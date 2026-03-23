"""
setup.py for srfm Python package.

Builds the pybind11 C++ extension when a build tree is available,
otherwise installs the pure-Python fallback (python/srfm/__init__.py).

Build with C++ extension:
    pip install pybind11 eigen3-headers
    cmake -B build -DSRFM_BUILD_PYTHON=ON
    cmake --build build
    pip install -e python/

Install pure-Python fallback (no build required):
    pip install -e python/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup

# ---------------------------------------------------------------------------
# Attempt to build the C++ extension via pybind11.
# Falls back to no extensions if pybind11 or Eigen is not available.
# ---------------------------------------------------------------------------

extensions = []

try:
    import pybind11  # noqa: F401

    # Locate Eigen3 headers.
    _eigen_candidates = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "C:/vcpkg/installed/x64-windows/include/eigen3",
        str(Path(__file__).parent.parent / "third_party" / "eigen"),
    ]
    _eigen_include = next(
        (p for p in _eigen_candidates if Path(p, "Eigen", "Dense").exists()),
        None,
    )

    _src_root = str(Path(__file__).parent.parent)
    _include_dirs = [
        pybind11.get_include(),
        str(Path(__file__).parent.parent / "include"),
    ]
    if _eigen_include:
        _include_dirs.append(_eigen_include)

    _extra_compile = ["-std=c++20", "-O3", "-DNDEBUG"]
    if sys.platform == "win32":
        _extra_compile = ["/std:c++20", "/O2", "/DNDEBUG", "/EHsc"]

    extensions = [
        Extension(
            name="srfm._srfm",
            sources=[
                "srfm/bindings.cpp",
                str(Path(_src_root) / "src" / "multi_asset.cpp"),
            ],
            include_dirs=_include_dirs,
            language="c++",
            extra_compile_args=_extra_compile,
        )
    ]
    print("[srfm] Building C++ extension.")
except ImportError:
    print("[srfm] pybind11 not found — installing pure-Python fallback only.")

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

_here = Path(__file__).parent

setup(
    name="srfm",
    version="1.1.0",
    description="Special Relativity in Financial Modeling — Python interface",
    long_description=(_here.parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Matthew Busel",
    license="MIT",
    url="https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling",
    packages=find_packages(where=str(_here)),
    package_dir={"": str(_here)},
    ext_modules=extensions,
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "full": ["numpy>=1.24", "pandas>=2.0", "scipy>=1.10", "matplotlib>=3.7"],
        "dev":  ["pytest>=7", "pybind11>=2.11", "mypy>=1.0"],
        "nb":   ["jupyterlab>=4", "ipywidgets"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=["finance", "special-relativity", "lorentz", "quantitative", "physics"],
)
