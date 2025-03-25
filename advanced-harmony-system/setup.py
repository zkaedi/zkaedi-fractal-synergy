from setuptools import setup, find_packages

setup(
    name="advanced_harmony",
    version="0.1.0",
    description="Advanced fractal, symbolic, and harmony agent system integration",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "matplotlib",
        "sympy",
    ],
    python_requires=">=3.7",
)
