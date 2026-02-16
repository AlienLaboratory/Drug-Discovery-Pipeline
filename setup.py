from setuptools import setup, find_packages

setup(
    name="claudedd",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "claudedd=claudedd.cli.main:main",
        ],
    },
)
