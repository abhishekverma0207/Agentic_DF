"""
setup.py for the DIQ Agentic_Forecasting package.

Lets us pip-install the repo into the cluster's system Python so
`utils`, `config`, and `crews` are natively importable from every
Python interpreter on every node — including pre-warmed Ray
workers, Spark executors, notebook kernels, etc.

The init script does:

    pip install --no-deps -e /Workspace/Users/.../Agentic_ForecastingGit

After that the closures we hand to Ray can do `from utils.model_training
import ...` without any working_dir / py_modules / PYTHONPATH gymnastics.
This is the single most reliable way to ship the codebase to workers
on a Databricks Ray cluster.

Why `find_namespace_packages` not `find_packages`:
    The repo uses PEP 420 implicit namespace packages — i.e. utils/,
    config/, and crews/ don't have __init__.py files.  `find_packages`
    skips directories without __init__.py; `find_namespace_packages`
    discovers them.  We add explicit `include=` patterns so test/
    script directories don't accidentally get bundled.

Why `install_requires=[]`:
    Dependencies are managed by the cluster init script's explicit
    pip pins.  Listing them here would re-resolve and could conflict
    with the careful pinning done in the init script (e.g. the user
    pins ray==2.9.0 to match the DBR-shipped Ray; pip resolution
    here could try to upgrade it).  Trust the init script.
"""
from setuptools import setup, find_namespace_packages

setup(
    name="diq_feu_agentic_forecasting",
    version="0.1.0",
    description=(
        "DIQ FEU Agentic Forecasting Upgrade — internal package "
        "shipping utils/, config/, and crews/ as importable modules."
    ),
    long_description=(
        "Internal Unilever package for the FEU agentic forecasting "
        "pipeline.  Pip-installed in the cluster init script so Ray "
        "workers, Spark executors, and notebook kernels can all import "
        "`utils.model_training`, `config.schema`, etc. natively without "
        "PYTHONPATH or runtime_env tricks."
    ),
    long_description_content_type="text/plain",
    python_requires=">=3.10",
    packages=find_namespace_packages(
        include=[
            "utils", "utils.*",
            "config", "config.*",
            "crews", "crews.*",
        ],
        exclude=[
            "tests", "tests.*",
            "notebooks", "notebooks.*",
            "**/__pycache__", "**/__pycache__.*",
        ],
    ),
    # NOTE: deliberately empty — see module docstring.  Deps are pinned
    # by the cluster init script.
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
)
