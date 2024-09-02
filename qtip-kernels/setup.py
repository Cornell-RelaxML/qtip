from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="qtip_kernels_cuda",
    ext_modules=[
        CUDAExtension(
            name="qtip_kernels",
            sources=[
                "src/wrapper.cpp", "src/inference.cu", "src/qtip_torch.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3", "--fast-math", "-lineinfo", "-std=c++17"],
                "nvcc": [
                    "-O3", "--use_fast_math", "-lineinfo", "-keep",
                    "-std=c++17", "--ptxas-options=-v"
                ],  #, "-gencode", "arch=compute_89,code=sm_89", "-gencode", "arch=compute_86,code=sm_86", "-gencode", "arch=compute_90,code=sm_90"],
            })
    ],
    cmdclass={"build_ext": BuildExtension},
)
