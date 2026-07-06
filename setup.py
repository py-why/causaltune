from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="causaltune",
    version="0.1.6",
    description="AutoML for Causal Inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Wise",
    url="https://github.com/py-why/causaltune",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10,<3.13",
    install_requires=[
        # Tightly-coupled / version-sensitive: pinned exactly.
        "dowhy==0.14",
        "econml==0.16.0",
        "FLAML==2.6.0",
        "xgboost==2.1.4",
        "numpy==2.2.6",
        # Loosely-coupled: floors (with caps where an upstream ceiling is real).
        "pandas>=2,<3",
        "scikit_learn>=1.4,<1.7",
        "category_encoders>=2.6.3",
        "pytest",
        "matplotlib",
        "dcor",
        "holidays",
        "wise-pizza",
        "seaborn",
    ],
    extras_require={
        "test": [
            "autoflake",
            "black==23.3.0",
            "flake8",
            "isort",
            "pytest",
            "pytest-cov",
            "nbmake",
        ],
        "ray": ["ray[tune]>=2.9"],
    },
    packages=find_packages(
        include=["causaltune", "causaltune.*"],
        exclude=["tests*"],
    ),
    include_package_data=True,
    keywords="causaltune",
)
