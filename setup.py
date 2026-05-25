from setuptools import find_packages, setup


setup(
    name="cross-term-greeks",
    version="0.1.0",
    description="PCA/SVD factor decomposition and cross-term Greek attribution utilities",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=["numpy>=1.23"],
)
