from setuptools import setup, Extension
from torch.utils import cpp_extension


setup (
    name = "moe_cuda_torch",
    ext_modules = [
        cpp_extension.CppExtension(
            "moe_cuda_torch",
            sources = ["python_api.cpp", "tk_bindings.cu"],
            py_limited_api=True
        ),
    ],
    
)