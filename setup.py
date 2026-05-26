from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


ext_modules = [
    Pybind11Extension(
        "cross_term_greeks._cpp",
        [
            "cpp/python/bindings.cpp",
            "cpp/src/attribution/CrossSecondOrderCalculator.cpp",
            "cpp/src/attribution/MatrixHessianCalculator.cpp",
            "cpp/src/attribution/ScalarSecondOrderCalculator.cpp",
            "cpp/src/decomposition/SVD/PCA.cpp",
            "cpp/src/decomposition/SVD/SVDInput.cpp",
            "cpp/src/numerics/FiniteDifference.cpp",
            "cpp/src/numerics/Interpolation.cpp",
            "cpp/src/numerics/QuadraticForm.cpp",
            "cpp/src/rates/Bootstrap.cpp",
            "cpp/src/rates/CurveCalibration.cpp",
            "cpp/src/rates/DiscountCurve.cpp",
            "cpp/src/rates/LmmSofr.cpp",
            "cpp/src/rates/SwaptionVolProxy.cpp",
            "cpp/src/scenarios/ScenarioLattice.cpp",
        ],
        include_dirs=["cpp/include"],
        cxx_std=20,
    )
]


setup(
    name="cross-term-greeks",
    version="0.1.0",
    description="PCA/SVD factor decomposition and cross-term Greek attribution utilities",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=["numpy>=1.23"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
