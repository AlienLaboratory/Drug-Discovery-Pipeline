from setuptools import setup, find_packages

setup(
    name="drugflow",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "drugflow=drugflow.cli.main:main",
        ],
    },
)
