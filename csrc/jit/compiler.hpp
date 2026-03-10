#pragma once
// JIT compiler
#include <runtime/format.hpp>
#include <nvrtc.h>
#include <moe_cuda/error.hpp>
#include <runtime/utils.h>
#include <jit/utils/files.hpp>
#include <jit/utils/math.hpp>
#include <runtime/device.hpp>
#include <jit/runtime.hpp>
#include <jit/cache.hpp>
#include <jit/utils/lazy_init.hpp>
#include <fstream>
#include <regex>
// Returns -isystem flags for the GCC C++ stdlib include paths, by running
// `gcc -v -x c++ /dev/null -fsyntax-only` and parsing its include search list.
// Returns -I flags for the GCC C++ stdlib include paths, filtered to standard
// system locations only (skips module-system or environment-injected paths).
static std::string get_gcc_system_include_flags() {
    auto [exit_code, output] = run_command("gcc -v -x c++ /dev/null -fsyntax-only 2>&1");
    if (exit_code != 0 && output.empty()) return "";

    std::string flags;
    bool in_search = false;
    std::istringstream ss(output);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.find("#include <...> search starts here") != std::string::npos) {
            in_search = true;
            continue;
        }
        if (in_search) {
            if (line.find("End of search list") != std::string::npos) break;
            auto start = line.find_first_not_of(" \t");
            if (start == std::string::npos) continue;
            auto path = line.substr(start);
            // skip paths injected by module systems or environments —
            // only keep standard system locations
            if (path.rfind("/usr", 0) != 0 && path.rfind("/lib", 0) != 0) continue;
            flags += fmt::format("-I{} ", path);
        }
    }
    return flags;
}

class Compiler {
    public :
        // root path of the repository
        inline static fs::path library_root_path;
        inline static fs::path library_include_path;
        inline static fs::path cuda_home;
        inline static std::string library_version;

        static std::string get_library_version() {
            std::vector<char> buffer;

            for (const auto& entry : all_files_in_dir(library_include_path / "moe_cuda")) {
                std::ifstream stream(entry, std::ios::binary);
                HOST_ASSERT(stream.is_open(), "file not open");
                buffer.insert(buffer.end(), 
                    std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
            }
            return get_hex_digest(buffer);
        }

        static void init_static_vars(
            fs::path library_root_path,
            fs::path cuda_home 
        ) {
            Compiler::library_root_path = library_root_path;
            Compiler::cuda_home = cuda_home;
            KernelRuntime::cuda_home = cuda_home;
            const auto include_override = get_env<std::string>("LIBRARY_INCLUDE_PATH", "");
            if (!include_override.empty()) {
                library_include_path = include_override;
            } else {
                library_include_path = library_root_path / "include";
            }
            library_version = get_library_version();
        }
        
        // compiler specific variables
        std::string signature; // overriden in subclasses, ex: 'nvcc'
        std::string flags;
    fs::path cache_dir_path;
        Compiler() {
            HOST_ASSERT(!library_root_path.empty(), "");
            HOST_ASSERT(!library_include_path.empty(), "");
            HOST_ASSERT(!cuda_home.empty(), "");
            HOST_ASSERT(!library_version.empty(), "");

            cache_dir_path = fs::path(get_env<std::string>("HOME")) / ".moe_cuda";
            if (const auto & manual_cache_dir = get_env<std::string>("JIT_CACHE_DIR"); !manual_cache_dir.empty()) {
                cache_dir_path = manual_cache_dir;
            }
            signature = "unknown";
            flags = fmt::format("-std=c++{} --diag-suppress=177 --ptxas-options=--register-usage-level=10",
                                        get_env<int>("JIT_CPP_STD", 20));
            // flags += " -Xcompiler -rdynamic -lineinfo";
        };

        virtual ~Compiler() = default;

        /*
        Given the current name and code from the runtime, this method searches the jit cache to see if the KernelRuntime object
        already exists. Otherwise it compiles with NVCC and then adds into the cache
        */
        fs::path make_tmp_dir() const {
            return make_dir(cache_dir_path / "tmp");
        }

        fs::path make_tmp_path() const {
            return make_tmp_dir() / get_uuid();
        }

        // write to a file
        void write_file(const fs::path& file_name, const std::string& data) const {
            const fs::path tmp_file_path = make_tmp_path();

            std::ofstream out(tmp_file_path, std::ios::binary);
            HOST_ASSERT(out.is_open(), fmt::format("Failed to open temporary file for writing: {}", tmp_file_path.string()).c_str());

            out.write(data.data(), data.size());
            HOST_ASSERT(out.good(), fmt::format("Failed to write data to temporary file: {}. Stream state: fail={}, bad={}", 
                                                 tmp_file_path.string(), out.fail(), out.bad()).c_str());
            out.close();

            std::error_code ec;
            fs::rename(tmp_file_path, file_name, ec);
        }

        std::shared_ptr<KernelRuntime> build(
            const std::string& name, const std::string& code
        ) const {
            // first create the kernel signature, which is hashed
            const auto kernel_signature = fmt::format("{}$${}$${}$${}$${}",
                name, library_version, signature, flags, code);
            const auto kernel_dir = fmt::format("kernel.{}.{}", 
                name, get_hex_digest(kernel_signature));
            
            const fs::path full_path = cache_dir_path / "cache" / kernel_dir;

            if (const auto& runtime_ptr = jit_cache->get_runtime(full_path); runtime_ptr != nullptr) {
                return runtime_ptr;
            }
            // if not in cache, we need to set up a new cubin and compile
            make_dir(full_path);
            const auto tmp_path = make_tmp_path();

            compile(code, full_path, tmp_path);

            // once write into tmp_path is complete, with a fully written binary, we need to rename it

            fs::rename(tmp_path, full_path / "kernel.cubin");
            // also need to populate with kernel.cu

            std::shared_ptr<KernelRuntime> new_runtime = std::make_shared<KernelRuntime>(full_path);
            jit_cache->store_runtime(full_path, new_runtime);

            return new_runtime;
        }

        virtual void compile(const std::string& code, const fs::path kernel_dir_path, const fs::path tmp_cubin_path) const = 0;
};



class NVCC_Compiler : public Compiler {

    fs::path nvcc_path;
    public:
    NVCC_Compiler() {
        // Initialize nvcc_path from cuda_home
        nvcc_path = cuda_home / "bin" / "nvcc";

        // get_nvcc_version
        auto version_cmd = fmt::format("{} --version", nvcc_path.string());
        auto [exitcode, output] = run_command(version_cmd);
        HOST_ASSERT(exitcode == 0, "Error when fetching the nvcc version");
        // use regex search to extract version

        std::smatch match; // set match
        HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")), "No nvcc version found");
        int major, minor;
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        HOST_WARNING(major == 12 && minor >= 9, "Warning: moe_cuda was built with NVCC 12.9, use 12.9 for better performance");
        signature = fmt::format("NVCC-{}-{}", major, minor);
        const auto arch = device_prop->get_major_minor();

        // ThunderKittens include paths: TK_ROOT override, then common repo-relative candidates.
        std::string extra_includes;
        auto tk_root_env = get_env<std::string>("TK_ROOT", "");
        std::vector<fs::path> tk_candidates;
        if (!tk_root_env.empty()) {
            tk_candidates.emplace_back(tk_root_env);
        }
        tk_candidates.push_back(library_root_path / "third-party" / "ThunderKittens");
        tk_candidates.push_back(library_root_path.parent_path() / "third-party" / "ThunderKittens");
        for (const auto& tk_root : tk_candidates) {
            if (fs::exists(tk_root / "include" / "kittens.cuh") &&
                fs::exists(tk_root / "prototype" / "prototype.cuh")) {
                extra_includes = fmt::format("-I{} -I{} -DKITTENS_HOPPER ",
                    (tk_root / "include").string(), (tk_root / "prototype").string());
                break;
            }
        }
        const auto gcc_system_includes = get_gcc_system_include_flags();
        auto [gcc_exit, gcc_path] = run_command("which gcc");
        if (gcc_exit == 0 && !gcc_path.empty()) {
            gcc_path.erase(gcc_path.find_last_not_of(" \t\n\r") + 1);
        }
        const auto compiler_bindir = (gcc_exit == 0 && !gcc_path.empty())
            ? fmt::format("--compiler-bindir={} ", gcc_path) : "";

        flags = fmt::format("{} -I{} {} {} {} -arch=sm_{} "
                                "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-abi "
                                "-cubin -O3 --expt-relaxed-constexpr --expt-extended-lambda",
                                flags, library_include_path.c_str(), extra_includes,
                                gcc_system_includes, compiler_bindir, device_prop->get_arch(true));
    };

    void compile(const std::string& code, const fs::path kernel_dir_path, const fs::path tmp_cubin_path) const override {

        // put kernel code in first, which is just the function pointer
        const fs::path code_path = kernel_dir_path / "kernel.cu";

        write_file(code_path, code);
        // compile into tmp_cubin_path, and then rename into orig
        std::string command = fmt::format("{} {} -o {} {}", nvcc_path.string(), code_path.string(), tmp_cubin_path.string(), flags);

        if (get_env<int>("JIT_DEBUG", 0) > 0) {
            printf("Compiling JIT runtime with NVCC options: ");
            printf("%s", command.c_str());
            printf("\n");
        }
        auto [exit_code, output] = run_command(command);
        if (exit_code != 0) {
            printf("NVCC compilation for file %s failed, with output: %s", code_path.string().c_str(), output.c_str());
            HOST_ASSERT(false, "");
        }

        if (get_env("JIT_DEBUG", 0)) {
            printf("%s", output.c_str());
        }
    }
};

class NVRTCCompiler final : public Compiler {
public:
    NVRTCCompiler () {
        int major, minor;
        NVRTC_CHECK(nvrtcVersion(&major, &minor));
        signature = fmt::format("NVRTC{}.{}", major, minor);
        HOST_ASSERT( (major > 12) || (major == 12 && minor >= 3), "NVRTC version must be at least 12.3");

        std::string include_dirs;
        include_dirs += fmt::format("-I{} ", library_include_path.string());
        include_dirs += fmt::format("-I{} ", (cuda_home / "include").string());
        include_dirs += get_gcc_system_include_flags();
        include_dirs += "-DKITTENS_HOPPER ";

        // ThunderKittens include paths (kittens.cuh + prototype.cuh):
        // TK_ROOT override, then common repo-relative candidates.
        auto tk_root_env = get_env<std::string>("TK_ROOT", "");
        std::vector<fs::path> tk_candidates;
        if (!tk_root_env.empty()) {
            tk_candidates.emplace_back(tk_root_env);
        }
        tk_candidates.push_back(library_root_path / "third-party" / "ThunderKittens");
        tk_candidates.push_back(library_root_path.parent_path() / "third-party" / "ThunderKittens");

        bool tk_found = false;
        for (const auto& tk_root : tk_candidates) {
            if (fs::exists(tk_root / "include" / "kittens.cuh") &&
                fs::exists(tk_root / "prototype" / "prototype.cuh")) {
                include_dirs += fmt::format("-I{} ", (tk_root / "include").string());
                include_dirs += fmt::format("-I{} ", (tk_root / "prototype").string());
                tk_found = true;
                break;
            }
        }
        if (!tk_found && get_env<int>("JIT_DEBUG", 0)) {
            printf("Warning: ThunderKittens headers not found for NVRTC include paths\n");
        }

        // Add PCH support for 12.8 and above
        std::string pch_flags;
        // DISABLED: PCH compilation hangs - investigate later
        if ((major > 12) || (major == 12 && minor >= 8)) {
            pch_flags = "--pch ";
            if (get_env<int>("JIT_DEBUG", 0)) {
                pch_flags += "--pch-verbose=true ";
            }
        }

        flags = fmt::format("{} {} --gpu-architecture=sm_{} -default-device {} --diag-suppress=639", flags, include_dirs, 
            device_prop->get_arch(major > 12 || minor >= 9), pch_flags);
    }
    
    void compile(const std::string& code, const fs::path kernel_dir_path, const fs::path tmp_cubin_path) const override{
        const fs::path code_path = kernel_dir_path / "kernel.cu";
        write_file(code_path, code);
        
        std::istringstream iss(flags);
        std::vector<std::string> options;
        std::string option;
        while (iss >> option) options.push_back(option);

        std::vector<const char *> option_cstrs;

        for (const auto& o : options) {
            option_cstrs.push_back(o.c_str());
        }

        if (get_env<int>("JIT_DEBUG", 0)) {
            printf("Compiling JIT runtime with NVRTC options: ");
            for (const auto& opt : option_cstrs) {
                printf("%s ", opt);
            }
            printf("\n");
        }

        nvrtcProgram program;
        NVRTC_CHECK(nvrtcCreateProgram(&program, code.c_str(), "kernel.cu", 0, nullptr, nullptr));
        const auto& compile_result = 
            nvrtcCompileProgram(program, static_cast<int>(option_cstrs.size()), option_cstrs.data());
        
        size_t log_size;
        NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));
        if (get_env("JIT_DEBUG", 0) || compile_result != NVRTC_SUCCESS) {
            if (compile_result != NVRTC_SUCCESS) {
                HOST_ASSERT(log_size > 1, "Log must be meaningful on errors");
            }
            if (log_size > 1) {
                // preallocate with log_size null characters
                std::string compilation_log(log_size, '\0');
                NVRTC_CHECK(nvrtcGetProgramLog(program, compilation_log.data()));
                printf("NVRTC log: %s", compilation_log.c_str());
            }
        }
        size_t cubin_size;
        NVRTC_CHECK(nvrtcGetCUBINSize(program, &cubin_size));
      
        std::string cubin(cubin_size, '\0');
        NVRTC_CHECK(nvrtcGetCUBIN(program, cubin.data()));
        write_file(tmp_cubin_path, cubin);
      
        NVRTC_CHECK(nvrtcDestroyProgram(&program));
    }
};

// init compiler

static auto compiler = LazyInit<Compiler> ([]() -> std::shared_ptr<Compiler> {
    if (get_env<int>("JIT_USE_NVRTC", 0)) {
        return std::make_shared<NVRTCCompiler>();
    }
    else {
        return std::make_shared<NVCC_Compiler>();
    }
});
