"""
Setup script for Brazilian REH Analyzer
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

# Filter out development dependencies
core_requirements = []
for req in requirements:
    if not any(
        dev_dep in req.lower()
        for dev_dep in ["pytest", "black", "flake8", "mypy", "jupyter", "plotly"]
    ):
        core_requirements.append(req)

setup(
    name="brazilian-reh-analyzer",
    version="1.0.0",
    author="KoscheiiB",
    author_email="KoscheiiB@users.noreply.github.com",
    description="Econometric analysis tool for assessing Brazilian Focus Bulletin inflation forecast rationality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KoscheiiB/brazilian-reh-analyzer",
    project_urls={
        "Bug Reports": "https://github.com/KoscheiiB/brazilian-reh-analyzer/issues",
        "Source": "https://github.com/KoscheiiB/brazilian-reh-analyzer",
        "Documentation": "https://github.com/KoscheiiB/brazilian-reh-analyzer/blob/main/README.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Economics",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "visualization": [
            "jupyter>=1.0.0",
            "plotly>=5.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "jupyter>=1.0.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "reh-analyzer=brazilian_reh_analyzer.__main__:main",
        ],
    },
    keywords=[
        "economics",
        "econometrics",
        "rational expectations",
        "forecasting",
        "brazil",
        "inflation",
        "central bank",
        "focus bulletin",
        "monetary policy",
    ],
    include_package_data=True,
    zip_safe=False,
)
