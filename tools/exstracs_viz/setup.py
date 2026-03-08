from setuptools import setup, find_packages

setup(
    name="exstracs_viz",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "exstracs-viz=exstracs_viz.cli:main",
        ],
    },
    author="Antigravity",
    description="Visualization package for ExSTraCS training metrics and statistics",
    python_requires=">=3.7",
)
