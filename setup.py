import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Core metadata
setup_kwargs = dict(
    name="graviton",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "csrc*"]),
)

# Compile C++ extensions ONLY on macOS (with MPS support)
if sys.platform == "darwin":
    try:
        setup_kwargs["ext_modules"] = [
            CppExtension(
                name="graviton_c",
                sources=["csrc/extension.mm"],
                extra_compile_args=["-std=c++17", "-fobjc-arc"],
                extra_link_args=["-framework", "Metal", "-framework", "Foundation"],
            )
        ]
        setup_kwargs["cmdclass"] = {"build_ext": BuildExtension}
        print("Metal extension generation enabled for macOS Darwin.")
    except Exception as e:
        print(f"Warning: Failed to setup CppExtension. Running in pure Python mode. Error: {e}")

if __name__ == "__main__":
    setup(**setup_kwargs)
