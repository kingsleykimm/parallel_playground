#pragma once
#include <jit/utils/culib.hpp>
#include <jit/utils/files.hpp>
#include <cuda.h>
#include <runtime/format.hpp>
#include <string>
#include <memory>
#include <regex>
class KernelRuntime {

    public:
    inline static fs::path cuda_home;

    KernelHandle kernel_handle;
    LibraryHandle library;

    // kernel runtime is created from the current cached_dir, which contains the compiled cuda binary that is converted into a CuFunction
    KernelRuntime(const fs::path cached_dir) {
        HOST_ASSERT(!cuda_home.empty(), "");
        // the directory should now contain the kernel.cu and kernel.cubin files
        // we need to load in the kernel to the kernel_handle, and the libraryahndle
        const fs::path cubin_path = cached_dir / "kernel.cubin";
        const fs::path cuobjdump_path = cuda_home / "bin" / "cuobjdump";
        // need to extract the symbol name of the function that is the kernel entrypoint, and extract that        
        const std::vector<std::string> filters = {"vprintf", "__instantiate_kernel", "__internal", "__assertfail"};
        auto command = fmt::format("{} -symbols {}", cuobjdump_path.string().c_str(), cubin_path.string().c_str());
        const auto& [exit_code, symbols] = run_command(command);
        if (exit_code != 0) {
            printf("Error in cuobjdump for file %s", cubin_path.string().c_str());
            HOST_ASSERT(false, "");
        }
        std::istringstream stream(symbols);

        std::vector<std::string> symbol_names;

        // STT_FUNC = function
        // STO_ENTRY = __global__ function entrypoint
        for (std::string line; std::getline(stream, line);) {
            if (line.find("STT_FUNC") != std::string::npos && line.find("STO_ENTRY") != std::string::npos &&
            std::none_of(filters.begin(), filters.end(), [&](const auto& name) {
                // condition to check that none of the filters are inside the symbol name
                return line.find(name) != std::string::npos;
            })) {
                // find the symbol name (last column of space-delimited table)
                auto pos = line.rfind(" ");
                if (pos != std::string::npos) {
                    symbol_names.push_back(line.substr(pos + 1));
                }
            }
        }
        HOST_ASSERT(symbol_names.size() > 0, "Symbol names were not found in cuobjdump");
        CUDA_CHECK(cuModuleLoad(&library, cubin_path.c_str()));
        CUDA_CHECK(cuModuleGetFunction(&kernel_handle, library, symbol_names[0].c_str()));
    }

    static bool contains_files(const fs::path kernel_dir_path) {
        return fs::exists(kernel_dir_path / "kernel.cu")
            && fs::exists(kernel_dir_path / "kernel.cubin");
    }

    ~KernelRuntime() = default;
};

// LaunchRuntime uses the CRTP (Curiously Recurring Template Pattern)
// Derived subclasses of LaunchRuntime will pass themselves in as the template
// this allows compile-time polymorphism, since derived class implement the exposed base methods
template<class Derived>
class LaunchRuntime {

    public:

        template <typename Args>
        static std::string generate(const Args& args) {
            const auto code = Derived::generate_impl(args);
            return code;
        }

        // this is a wrapper function around the args, pass down the actual implementation logic to the derived
        template<typename Args>
        static void launch(
            std::shared_ptr<KernelRuntime>& kernel_runtime,
            const Args& args
        ) {
            // construct launch_config here
            // construct the launch config here
           
            LaunchConfigHandle launch_handle = create_launch_config(kernel_runtime->kernel_handle, args.launch_config.smem_size,
                args.launch_config.blockDim, args.launch_config.gridDim, args.launch_config.num_multicast, args.launch_config.stream);
            Derived::launch_impl(kernel_runtime->kernel_handle, launch_handle, args);
        }

};