#pragma once

#include <cstring>
#include <cuda.h>
#include <moe_cuda/error.hpp>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

enum class CUMemHandleKind {
  Local,
  FileDescriptor,
  Fabric,
};

// Forward declarations
class CUMemMapping;
class CUMemImportHandle;

// RAII wrapper around a CUmem virtual memory mapping
class CUMemMapping {
public:
  CUMemMapping() : ptr_(0), alloc_size_(0), device_id_(0), mapped_(false) {}

  CUMemMapping(CUdeviceptr ptr, size_t size, int device_id)
      : ptr_(ptr), alloc_size_(size), device_id_(device_id), mapped_(true) {}

  ~CUMemMapping() { release(); }

  // Move only
  CUMemMapping(CUMemMapping &&other) noexcept
      : ptr_(other.ptr_), alloc_size_(other.alloc_size_),
        device_id_(other.device_id_), mapped_(other.mapped_) {
    other.mapped_ = false;
    other.ptr_ = 0;
  }

  CUMemMapping &operator=(CUMemMapping &&other) noexcept {
    if (this != &other) {
      release();
      ptr_ = other.ptr_;
      alloc_size_ = other.alloc_size_;
      device_id_ = other.device_id_;
      mapped_ = other.mapped_;
      other.mapped_ = false;
      other.ptr_ = 0;
    }
    return *this;
  }

  CUMemMapping(const CUMemMapping &) = delete;
  CUMemMapping &operator=(const CUMemMapping &) = delete;

  void *data_ptr() const { return (void *)ptr_; }
  size_t size() const { return alloc_size_; }
  int device_id() const { return device_id_; }

  void unmap() {
    if (mapped_) {
      CUDA_CHECK(cuMemUnmap(ptr_, alloc_size_));
      mapped_ = false;
    }
  }

private:
  void release() {
    if (mapped_) {
      cuMemUnmap(ptr_, alloc_size_);
    }
    if (ptr_) {
      cuMemAddressFree(ptr_, alloc_size_);
    }
  }

  CUdeviceptr ptr_;
  size_t alloc_size_;
  int device_id_;
  bool mapped_;
};

// Serializable export data for cross-process sharing
struct CUMemExportData {
  int fd;
  uint32_t pid;
  size_t alloc_size;
  int device_id;
};

// RAII wrapper around a CUmem physical allocation
class CUMemAllocHandle {
public:
  CUMemAllocHandle()
      : alloc_size_(0), device_id_(-1), handle_kind_(CUMemHandleKind::Local),
        handle_(0) {}

  CUMemAllocHandle(size_t size, int device_id, CUMemHandleKind handle_kind)
      : alloc_size_(0), device_id_(device_id), handle_kind_(handle_kind),
        handle_(0) {

    CUmemAllocationProp props = {};
    props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    props.location.id = device_id;
    props.requestedHandleTypes = to_cu_handle_type(handle_kind);
    props.allocFlags.gpuDirectRDMACapable = 1;

    size_t granularity = 0;
    CUDA_CHECK(cuMemGetAllocationGranularity(&granularity, &props,
                                             CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    alloc_size_ = ((size + granularity - 1) / granularity) * granularity;

    CUDA_CHECK(cuMemCreate(&handle_, alloc_size_, &props, 0));
  }

  ~CUMemAllocHandle() {
    if (handle_) {
      cuMemRelease(handle_);
    }
  }

  // Move only
  CUMemAllocHandle(CUMemAllocHandle &&other) noexcept
      : alloc_size_(other.alloc_size_), device_id_(other.device_id_),
        handle_kind_(other.handle_kind_), handle_(other.handle_) {
    other.handle_ = 0;
  }

  CUMemAllocHandle &operator=(CUMemAllocHandle &&other) noexcept {
    if (this != &other) {
      if (handle_)
        cuMemRelease(handle_);
      alloc_size_ = other.alloc_size_;
      device_id_ = other.device_id_;
      handle_kind_ = other.handle_kind_;
      handle_ = other.handle_;
      other.handle_ = 0;
    }
    return *this;
  }

  CUMemAllocHandle(const CUMemAllocHandle &) = delete;
  CUMemAllocHandle &operator=(const CUMemAllocHandle &) = delete;

  // Map this allocation into a virtual address, accessible from
  // `access_device_id`
  CUMemMapping map(int access_device_id) const {
    CUdeviceptr ptr = 0;
    CUDA_CHECK(cuMemAddressReserve(&ptr, alloc_size_, 0, 0, 0));
    CUDA_CHECK(cuMemMap(ptr, alloc_size_, 0, handle_, 0));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = access_device_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDA_CHECK(cuMemSetAccess(ptr, alloc_size_, &access, 1));

    return CUMemMapping(ptr, alloc_size_, access_device_id);
  }

  // Export as shareable data for cross-process transfer (via MPI, etc.)
  CUMemExportData export_handle() const {
    HOST_ASSERT(handle_kind_ == CUMemHandleKind::FileDescriptor,
                "Can only export FileDescriptor handles");

    int fd = -1;
    CUDA_CHECK(cuMemExportToShareableHandle(
        &fd, handle_, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

    return CUMemExportData{
        .fd = fd,
        .pid = (uint32_t)getpid(),
        .alloc_size = alloc_size_,
        .device_id = device_id_,
    };
  }

  size_t size() const { return alloc_size_; }
  int device_id() const { return device_id_; }

private:
  static CUmemAllocationHandleType_enum
  to_cu_handle_type(CUMemHandleKind kind) {
    switch (kind) {
    case CUMemHandleKind::Local:
      return CU_MEM_HANDLE_TYPE_NONE;
    case CUMemHandleKind::FileDescriptor:
      return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    case CUMemHandleKind::Fabric:
      return CU_MEM_HANDLE_TYPE_FABRIC;
    }
    return CU_MEM_HANDLE_TYPE_NONE;
  }

  size_t alloc_size_;
  int device_id_;
  CUMemHandleKind handle_kind_;
  CUmemGenericAllocationHandle handle_;
};

// Import handle from cross-process export data
class CUMemImportHandle {
public:
  // Import from a CUMemExportData received from another process
  // Uses pidfd_getfd to duplicate the file descriptor into this process
  static CUMemImportHandle from_export(const CUMemExportData &data) {
    // Open pidfd for the source process
    int pidfd = (int)syscall(SYS_pidfd_open, data.pid, 0);
    HOST_ASSERT(pidfd >= 0, "Failed to open pidfd for import");

    // Duplicate the fd from the source process into ours
    int local_fd = (int)syscall(SYS_pidfd_getfd, pidfd, data.fd, 0);
    close(pidfd);
    HOST_ASSERT(local_fd >= 0, "Failed to get fd from peer process");

    // Import into CUDA
    CUmemGenericAllocationHandle handle = 0;
    CUDA_CHECK(cuMemImportFromShareableHandle(
        &handle, (void *)(intptr_t)local_fd,
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

    close(local_fd);

    return CUMemImportHandle(handle, data.alloc_size, data.device_id);
  }

  ~CUMemImportHandle() {
    if (handle_) {
      cuMemRelease(handle_);
    }
  }

  // Move only
  CUMemImportHandle(CUMemImportHandle &&other) noexcept
      : handle_(other.handle_), alloc_size_(other.alloc_size_),
        device_id_(other.device_id_) {
    other.handle_ = 0;
  }

  CUMemImportHandle &operator=(CUMemImportHandle &&other) noexcept {
    if (this != &other) {
      if (handle_)
        cuMemRelease(handle_);
      handle_ = other.handle_;
      alloc_size_ = other.alloc_size_;
      device_id_ = other.device_id_;
      other.handle_ = 0;
    }
    return *this;
  }

  CUMemImportHandle(const CUMemImportHandle &) = delete;
  CUMemImportHandle &operator=(const CUMemImportHandle &) = delete;

  // Map this imported allocation, accessible from `access_device_id`
  CUMemMapping map(int access_device_id) const {
    CUdeviceptr ptr = 0;
    CUDA_CHECK(cuMemAddressReserve(&ptr, alloc_size_, 0, 0, 0));
    CUDA_CHECK(cuMemMap(ptr, alloc_size_, 0, handle_, 0));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = access_device_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDA_CHECK(cuMemSetAccess(ptr, alloc_size_, &access, 1));

    return CUMemMapping(ptr, alloc_size_, access_device_id);
  }

  size_t size() const { return alloc_size_; }

private:
  CUMemImportHandle(CUmemGenericAllocationHandle handle, size_t size,
                    int device_id)
      : handle_(handle), alloc_size_(size), device_id_(device_id) {}

  CUmemGenericAllocationHandle handle_;
  size_t alloc_size_;
  int device_id_;
};
