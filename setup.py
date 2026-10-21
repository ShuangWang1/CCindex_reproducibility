"""Build the current optimized C++ matcher."""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


setup(
    name="ccindex-matcher",
    version="0.1.0",
    description="Triangle-indexed spatial graph motif matching",
    ext_modules=[
        Pybind11Extension(
            "graph_pattern_cpp",
            ["graph_pattern_cpp.cpp"],
            extra_compile_args=["-O3", "-DNDEBUG"],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.11",
)
