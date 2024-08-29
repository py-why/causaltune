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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=[
        "dowhy==0.9.1",
        "econml==0.14.1",
        "FLAML==2.2.0",
        "xgboost==1.7.6",
        "numpy==1.23.5",
        "pandas",
        "pytest",
        "scikit_learn",
        "matplotlib",
        "dcor",
        "holidays",
        "setuptools==65.5.1",
        "wise-pizza",
        "seaborn",
        "category_encoders==2.6.3",
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
        "ray": ["ray[tune]~=1.11.0"],
    },
    packages=find_packages(
        include=["causaltune", "causaltune.*"],
        exclude=["tests*"],
    ),
    include_package_data=True,
    keywords="causaltune",
)
