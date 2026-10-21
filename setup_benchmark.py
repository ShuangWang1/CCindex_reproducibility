"""Build isolated old and current graph-matcher extensions for benchmarking."""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


COMMON_ARGS = ["-O3", "-DNDEBUG"]


ext_modules = [
    Pybind11Extension(
        "graph_pattern_cpp_reference",
        ["graph_pattern_cpp_reference.cpp"],
        define_macros=[("graph_pattern_cpp", "graph_pattern_cpp_reference")],
        extra_compile_args=COMMON_ARGS,
        language="c++",
        cxx_std=17,
    ),
    Pybind11Extension(
        "graph_pattern_cpp",
        ["graph_pattern_cpp.cpp"],
        define_macros=[("graph_pattern_cpp", "graph_pattern_cpp")],
        extra_compile_args=COMMON_ARGS,
        language="c++",
        cxx_std=17,
    ),
]


setup(
    name="graph-pattern-matcher-current-benchmark",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)
