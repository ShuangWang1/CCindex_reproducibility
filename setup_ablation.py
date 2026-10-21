from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


setup(
    name="graph-pattern-ablation-cpp",
    ext_modules=[
        Pybind11Extension(
            "graph_pattern_ablation_cpp",
            ["CCindex/Claude_sonnet4.5/graph_pattern_cpp.cpp"],
            define_macros=[("GRAPH_PATTERN_MODULE_NAME", "graph_pattern_ablation_cpp")],
            extra_compile_args=["-O3"],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
