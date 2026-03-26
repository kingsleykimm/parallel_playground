#pragma once
#include <jit/runtime.hpp>
#include <mutex>
// Kernel Runtime Cache
// Mirrors the structure of the JIT cache

class KernelRuntimeCache {
public:
  std::unordered_map<fs::path, std::shared_ptr<KernelRuntime>> cache;
  std::mutex cache_mutex;
  KernelRuntimeCache() = default;

  std::shared_ptr<KernelRuntime> get_runtime(const fs::path key) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto found = cache.find(key);
    if (found != cache.end()) {
      return found->second;
    }

    if (KernelRuntime::contains_files(key)) {
      cache[key] = std::make_shared<KernelRuntime>(key);
      return cache[key];
    }
    return nullptr;
  }

  void store_runtime(const fs::path key,
                     std::shared_ptr<KernelRuntime> &kernel_runtime) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (cache.find(key) != cache.end()) {
      return;
    }
    if (KernelRuntime::contains_files(key)) {
      cache[key] = kernel_runtime;
    }
  }
};

static auto jit_cache = std::make_shared<KernelRuntimeCache>();
