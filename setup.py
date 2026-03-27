import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from setuptools import Extension, setup, find_packages
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME
)

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False

_ROOT = Path(__file__).resolve().parent


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str) -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(BuildExtension):

    def build_extension(self, ext) -> None:
        if isinstance(ext, CMakeExtension):
            ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
            extdir = ext_fullpath.parent.resolve()

            build_temp = Path(self.build_temp) / ext.name
            build_temp.mkdir(parents=True, exist_ok=True)

            cmake_bin = os.environ.get("CMAKE") or shutil.which("cmake")
            if not cmake_bin:
                raise RuntimeError(
                    "cmake not found. Set CMAKE=/path/to/cmake or add cmake to PATH.")

            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get(
                "CUDA_HOME_PATH") or "/usr/local/cuda"

            cmake_args = [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_temp}",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                f"-DPython3_EXECUTABLE={sys.executable}",
                f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_home}",
                "-DBUILD_PYBIND=ON",
                "-DBUILD_TESTS=OFF",
                "-DFETCH_LIBTORCH=OFF",
                f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            ]

            build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")

            build_args = ["--config", build_type]
            if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
                build_args += ["-j4"]

            subprocess.run([cmake_bin, ext.sourcedir, *cmake_args],
                           cwd=build_temp, check=True)
            subprocess.run([cmake_bin, "--build", ".", *build_args],
                           cwd=build_temp, check=True)

            built_files = glob.glob(str(build_temp / "moe_cuda*.so"))
            if not built_files:
                raise RuntimeError(
                    f"Built .so file not found in {build_temp}. "
                    f"Available files: {list(build_temp.glob('*'))}"
                )

            extdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_files[0], ext_fullpath)

        else:
            super().build_extension(ext)


def get_torch_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"

    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x3090000",
            "-DTORCH_TARGET_VERSION=0x020a000000000000"
        ],
        "nvcc": [
            "-fdiagnostics-color=always",
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
            "-DUSE_CUDA"
        ]
    }

    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    torch_api_path = str(_ROOT / "csrc" / "torch_api.cpp")

    # Built by CMake (moe_cuda_lib STATIC) into build/moe_cuda_lib/libmoe_cuda_lib.a — see CMakeLists.txt.
    # CMakeExtension runs before this extension so the archive should exist at link time.
    moe_cuda_lib_a = _ROOT / "build" / "moe_cuda_lib" / "libmoe_cuda_lib.a"

    return extension(
        # Basename must match moe_cuda_python/__init__.py (glob moe_cuda_torch*.so).
        name="moe_cuda_python.moe_cuda_torch",
        sources=[torch_api_path],
        extra_objects=[str(moe_cuda_lib_a)],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        py_limited_api=py_limited_api,
    )


setup(
    name="moe_cuda",
    version="0.0.1",
    ext_modules=[
        # Pybind + CUDA: built by CMake (see CMakeLists BUILD_PYBIND).
        CMakeExtension("moe_cuda", sourcedir=str(_ROOT)),
        # IMPORTANT: CMake needs to be run first in order to generate a lib file that the torch extension can load in to get the actual implementations
        # torch.ops stable registrations only (torch_api.cpp).
        get_torch_extensions(),
    ],
    packages=find_packages(),
    install_requires=["torch>=2.10.0"],
    cmdclass={"build_ext": CMakeBuild},
    options=(
        {"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {}
    ),
    zip_safe=False,
)
