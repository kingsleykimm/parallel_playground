cmake_minimum_required(VERSION 3.22)
project(parallel_playground LANGUAGES CXX CUDA)

option(BUILD_TESTS "Build C++ tests" ON)
option(FETCH_LIBTORCH "Auto-fetch LibTorch when Torch is not found" OFF)
set(LIBTORCH_VERSION "2.10.0" CACHE STRING "LibTorch version to fetch")
set(LIBTORCH_CUDA_TAG "" CACHE STRING "Override CUDA tag (e.g. cu128)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "90a")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR})
endforeach()

include(FetchContent)
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cmake_deps" CACHE PATH "Persistent FetchContent cache")

find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
    set(TORCH_CUDA_ARCH_LIST "9.0a" CACHE STRING "Torch CUDA arch list")
endif()
message(STATUS "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")

set(HAS_VALID_TORCH_DIR OFF)
if(DEFINED Torch_DIR AND NOT "${Torch_DIR}" STREQUAL "" AND NOT "${Torch_DIR}" MATCHES "-NOTFOUND$")
    set(HAS_VALID_TORCH_DIR ON)
endif()

if(NOT HAS_VALID_TORCH_DIR)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE TORCH_CMAKE_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(TORCH_CMAKE_PATH)
            list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
            message(STATUS "Using PyTorch from Python environment: ${TORCH_CMAKE_PATH}")
        endif()
    endif()
endif()

find_package(Torch QUIET CONFIG)
if(NOT Torch_FOUND AND FETCH_LIBTORCH AND NOT HAS_VALID_TORCH_DIR)
    if(LIBTORCH_CUDA_TAG)
        set(_libtorch_cuda_tag "${LIBTORCH_CUDA_TAG}")
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION_MINOR GREATER_EQUAL 8)
        set(_libtorch_cuda_tag "cu128")
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION_MINOR EQUAL 6)
        set(_libtorch_cuda_tag "cu126")
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION_MINOR EQUAL 4)
        set(_libtorch_cuda_tag "cu124")
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 11 AND CUDAToolkit_VERSION_MINOR EQUAL 8)
        set(_libtorch_cuda_tag "cu118")
    else()
        message(FATAL_ERROR
            "No preconfigured LibTorch mapping for CUDA ${CUDAToolkit_VERSION}. "
            "Set -DLIBTORCH_CUDA_TAG=<cuXXX> manually.")
    endif()

    set(_libtorch_url "https://download.pytorch.org/libtorch/${_libtorch_cuda_tag}/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2B${_libtorch_cuda_tag}.zip")
    FetchContent_Declare(libtorch_prebuilt URL "${_libtorch_url}" DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_GetProperties(libtorch_prebuilt)
    if(NOT libtorch_prebuilt_POPULATED)
        FetchContent_MakeAvailable(libtorch_prebuilt)
    endif()
    set(Torch_DIR "${libtorch_prebuilt_SOURCE_DIR}/share/cmake/Torch" CACHE PATH "Torch CMake config path" FORCE)
    list(PREPEND CMAKE_PREFIX_PATH "${libtorch_prebuilt_SOURCE_DIR}")
    find_package(Torch REQUIRED CONFIG)
elseif(NOT Torch_FOUND)
    message(FATAL_ERROR "Torch was not found. Install torch in your env, set -DTorch_DIR, or enable -DFETCH_LIBTORCH=ON")
endif()

# Optional Torch-wheel NCCL discovery hint.
set(TORCH_NCCL_ROOT "")
if(Torch_FOUND AND DEFINED Torch_DIR)
    get_filename_component(_torch_cmake_dir "${Torch_DIR}" ABSOLUTE)
    get_filename_component(_torch_cmake_parent "${_torch_cmake_dir}" DIRECTORY)  # .../share/cmake
    get_filename_component(_torch_share_dir "${_torch_cmake_parent}" DIRECTORY)   # .../share
    get_filename_component(_torch_pkg_dir "${_torch_share_dir}" DIRECTORY)        # .../torch
    get_filename_component(_torch_site_dir "${_torch_pkg_dir}" DIRECTORY)         # .../site-packages
    set(_torch_nccl_candidate "${_torch_site_dir}/nvidia/nccl")
    if(EXISTS "${_torch_nccl_candidate}/include/nccl.h")
        set(TORCH_NCCL_ROOT "${_torch_nccl_candidate}")
        message(STATUS "Using Torch NCCL hint root: ${TORCH_NCCL_ROOT}")
    endif()
endif()

# CUTLASS
# set(SUBMODULE_CUTLASS_INCLUDE "${CMAKE_SOURCE_DIR}/third-party/cutlass/include")
# if(EXISTS "${SUBMODULE_CUTLASS_INCLUDE}/cutlass" AND EXISTS "${SUBMODULE_CUTLASS_INCLUDE}/cute")
#     set(CUTLASS_INCLUDE "${SUBMODULE_CUTLASS_INCLUDE}")
#     message(STATUS "Using CUTLASS submodule headers from ${CUTLASS_INCLUDE}")
# else()
#     FetchContent_Declare(
#         cutlass
#         GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
#         GIT_TAG v3.8.0
#         GIT_SHALLOW TRUE
#     )
#     FetchContent_GetProperties(cutlass)
#     if(NOT cutlass_POPULATED)
#         FetchContent_Populate(cutlass)
#     endif()
#     set(CUTLASS_INCLUDE ${cutlass_SOURCE_DIR}/include)
#     message(STATUS "Using FetchContent CUTLASS headers from ${CUTLASS_INCLUDE}")
# endif()

set (SUBMODULE_TK_INCLUDE "${CMAKE_SOURCE_DIR}/third-party/ThunderKittens/include")
set (SUBMODULE_TK_PROTOTYPE "${CMAKE_SOURCE_DIR}/third-party/ThunderKittens/prototype")
if (EXISTS ${SUBMODULE_TK_INCLUDE})
    set (TK_INCLUDE ${SUBMODULE_TK_INCLUDE})
    set (TK_PROTOTYPE_INCLUDE ${SUBMODULE_TK_PROTOTYPE})
    message(STATUS "Using ThunderKittens submodule headers from ${TK_INCLUDE}")
else()
    FetchContent_Declare(
        ThunderKittens
        GIT_REPOSITORY https://github.com/HazyResearch/ThunderKittens.git
        GIT_TAG main
    )
    FetchContent_GetProperties(ThunderKittens)
    if(NOT ThunderKittens_POPULATED)
        FetchContent_Populate(ThunderKittens)
    endif()
    set(TK_INCLUDE ${ThunderKittens_SOURCE_DIR}/include)
    set(TK_PROTOTYPE_INCLUDE ${ThunderKittens_SOURCE_DIR}/prototype)
    message(STATUS "Using FetchContent ThunderKittens headers from ${TK_INCLUDE}")
endif()

# fmt
set(SUBMODULE_FMT_DIR "${CMAKE_SOURCE_DIR}/third-party/fmt")
if(EXISTS "${SUBMODULE_FMT_DIR}/CMakeLists.txt")
    add_subdirectory("${SUBMODULE_FMT_DIR}" "${CMAKE_BINARY_DIR}/third_party_fmt" EXCLUDE_FROM_ALL)
else()
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 10.2.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(fmt)
endif()

# MPI (required)
set(MPI_SKIP_COMPILER_WRAPPER TRUE CACHE BOOL "Avoid MPI wrapper parsing issues in FindMPI")
find_package(MPI REQUIRED COMPONENTS CXX)
set(MOE_CUDA_USE_MPI ON)
message(STATUS "MPI found: enabling all2all support")
if(TARGET MPI::MPI_CXX)
    get_target_property(_mpi_inc_dirs MPI::MPI_CXX INTERFACE_INCLUDE_DIRECTORIES)
    if(_mpi_inc_dirs)
        set(_mpi_inc_dirs_clean "")
        foreach(_inc IN LISTS _mpi_inc_dirs)
            string(REGEX REPLACE "^-I" "" _inc_clean "${_inc}")
            list(APPEND _mpi_inc_dirs_clean "${_inc_clean}")
        endforeach()
        set_target_properties(MPI::MPI_CXX PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_mpi_inc_dirs_clean}")
    endif()
endif()

# NCCL (required)
set(_NCCL_HINT_ROOTS "")
if(DEFINED ENV{NCCL_ROOT} AND NOT "$ENV{NCCL_ROOT}" STREQUAL "")
    list(APPEND _NCCL_HINT_ROOTS "$ENV{NCCL_ROOT}")
endif()
if(TORCH_NCCL_ROOT)
    list(APPEND _NCCL_HINT_ROOTS "${TORCH_NCCL_ROOT}")
endif()

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ${_NCCL_HINT_ROOTS}
    PATH_SUFFIXES include
)
find_library(NCCL_LIBRARY
    NAMES nccl nccl.so.2 libnccl.so.2
    HINTS ${_NCCL_HINT_ROOTS}
    PATH_SUFFIXES lib lib64
)

if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    add_library(moe_cuda_nccl INTERFACE)
    target_include_directories(moe_cuda_nccl INTERFACE ${NCCL_INCLUDE_DIR})
    target_link_libraries(moe_cuda_nccl INTERFACE ${NCCL_LIBRARY})
    add_library(NCCL::NCCL ALIAS moe_cuda_nccl)
    message(STATUS "NCCL located via hinted/manual search")
else()
    find_package(NCCL QUIET)
    if(NCCL_FOUND)
        message(STATUS "NCCL package found")
        if(NOT TARGET NCCL::NCCL)
            add_library(moe_cuda_nccl INTERFACE)
            if(NCCL_INCLUDE_DIRS)
                target_include_directories(moe_cuda_nccl INTERFACE ${NCCL_INCLUDE_DIRS})
            elseif(NCCL_INCLUDE_DIR)
                target_include_directories(moe_cuda_nccl INTERFACE ${NCCL_INCLUDE_DIR})
            endif()
            if(NCCL_LIBRARIES)
                target_link_libraries(moe_cuda_nccl INTERFACE ${NCCL_LIBRARIES})
            elseif(NCCL_LIBRARY)
                target_link_libraries(moe_cuda_nccl INTERFACE ${NCCL_LIBRARY})
            endif()
            add_library(NCCL::NCCL ALIAS moe_cuda_nccl)
        endif()
    else()
        message(FATAL_ERROR "NCCL is required but was not found. Set NCCL_ROOT or install an NCCL package with CMake config.")
    endif()
endif()

add_library(moe_cuda_lib STATIC
    src/moe_ops.cu
    csrc/apis/moe_forward.cpp
    csrc/jit_kernels/fp8_tk_globals_factory.cu
)

target_include_directories(moe_cuda_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/moe_cuda/include
        ${CMAKE_SOURCE_DIR}/csrc
        # ${CUTLASS_INCLUDE}
        ${TK_INCLUDE}
        ${TK_PROTOTYPE_INCLUDE}
        ${TORCH_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(moe_cuda_lib PUBLIC
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvJitLink
    CUDA::cublas
    CUDA::cublasLt
    MPI::MPI_CXX
    NCCL::NCCL
    fmt::fmt
)

target_compile_definitions(moe_cuda_lib PUBLIC FMT_USE_INT128=0)

# ThunderKittens globals factory needs KITTENS_HOPPER for TMA/WGMMA support
set_source_files_properties(csrc/jit_kernels/fp8_tk_globals_factory.cu PROPERTIES
    COMPILE_DEFINITIONS "KITTENS_HOPPER"
)
target_compile_definitions(moe_cuda_lib PUBLIC MOE_CUDA_USE_MPI=1)

set_target_properties(moe_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(moe_cuda_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --diag-suppress=20208
        -Xcompiler=-fPIC
        --use_fast_math
        -lineinfo
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -fPIC
    >
)

if(BUILD_TESTS)
    enable_testing()

    set(BASE_TESTS
        test/test_cpp/test_bf16_gemm.cpp
        test/test_cpp/test_fp8.cpp
        test/test_cpp/test_kernel2.cpp
    )

    foreach(TEST_SRC ${BASE_TESTS})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SRC})
        set_source_files_properties(${TEST_SRC} PROPERTIES LANGUAGE CUDA)
        target_include_directories(${TEST_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/moe_cuda/include
            ${CMAKE_SOURCE_DIR}/csrc
            ${CMAKE_SOURCE_DIR}/test/test_cpp
            # ${CUTLASS_INCLUDE}
            ${TK_INCLUDE}
            ${TORCH_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(${TEST_NAME} PRIVATE
            moe_cuda_lib
            ${TORCH_LIBRARIES}
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nvrtc
            CUDA::nvJitLink
            CUDA::cublas
            CUDA::cublasLt
            fmt::fmt
        )
        target_compile_definitions(${TEST_NAME} PRIVATE FMT_USE_INT128=0)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()

    add_executable(test_all2all test/test_cpp/test_all2all.cpp)
    set_source_files_properties(test/test_cpp/test_all2all.cpp PROPERTIES LANGUAGE CUDA)
    target_include_directories(test_all2all PRIVATE
        ${CMAKE_SOURCE_DIR}/moe_cuda/include
        ${CMAKE_SOURCE_DIR}/csrc
        ${CMAKE_SOURCE_DIR}/test/test_cpp
        # ${CUTLASS_INCLUDE}
        ${TORCH_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(test_all2all PRIVATE
        moe_cuda_lib
        ${TORCH_LIBRARIES}
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
        CUDA::nvJitLink
        CUDA::cublas
        CUDA::cublasLt
        MPI::MPI_CXX
        NCCL::NCCL
        fmt::fmt
    )
    target_compile_definitions(test_all2all PRIVATE FMT_USE_INT128=0 MOE_CUDA_USE_MPI=1)
    add_test(NAME test_all2all COMMAND test_all2all)
endif()

install(TARGETS moe_cuda_lib ARCHIVE DESTINATION lib)
install(DIRECTORY moe_cuda/include/moe_cuda DESTINATION include)
install(DIRECTORY csrc/ DESTINATION include/moe_cuda_csrc FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh")
