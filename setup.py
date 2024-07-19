from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'generate_edges',
        ['data/graph_gen/generate_edges.cpp'],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name='gnn',
    ext_modules=ext_modules,
    zip_safe=False,
)
