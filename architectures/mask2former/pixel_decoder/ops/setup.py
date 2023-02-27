# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import os
import glob

import torch

from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    extension = CUDAExtension
    sources += source_cuda

    if CUDA_HOME is not None:
        gpu_platform = "CUDA"
        extra_compile_args["nvcc"] = []
    elif ROCM_HOME is not None:
        from torch.utils.hipify import hipify_python
        gpu_platform = "ROCM"
        hipify_python.hipify(
            project_directory=os.getcwd(),
            output_directory=os.getcwd(),
            header_include_dirs=[os.path.join(extensions_dir, 'cuda')],
            includes=[os.path.join(os.getcwd(), '*')],
            extra_files=[os.path.abspath(s) for s in sources],
            show_detailed=True,
            is_pytorch_extension=True,
            hipify_extra_files_only=True,
        )

        rocm_arch = os.getenv("ROCM_ARCH", None)
        if rocm_arch is not None:
            extra_compile_args["nvcc"] = [f"--offload-arch={rocm_arch}"]
    else:
        raise NotImplementedError('Cuda or ROCm are not available')

    define_macros += [(f"WITH_CUDA", None)]
    extra_compile_args["nvcc"] += [
        "-DCUDA_HAS_FP16=1",
        f"-D__{gpu_platform}_NO_HALF_OPERATORS__",
        f"-D__{gpu_platform}_NO_HALF_CONVERSIONS__",
        f"-D__{gpu_platform}_NO_HALF2_OPERATORS__",
    ]

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
