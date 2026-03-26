import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
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


setup(
    ext_modules=[CMakeExtension("moe_cuda")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
