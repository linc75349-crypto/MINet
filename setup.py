import os
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="minet",
    version="1.0.0",
    description="MINet: Multiscale Interactive Network for Real-Time Salient Object Detection of Strip Steel Surface Defects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kunye Shen, Xiaofei Zhou, Zhi Liu",
    url="https://github.com/Kunye-Shen/MINet",
    packages=find_packages(exclude=["figures", "model_save", "results", "Dataset"]),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.4.0",
        "torchvision>=0.5.0",
        "numpy>=1.18.1",
        "scikit-image>=0.16.0",
        "Pillow>=7.0.0",
        "tqdm>=4.42.0",
    ],
    extras_require={
        "dev": ["flake8"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
