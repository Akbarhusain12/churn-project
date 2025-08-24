from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path):
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() != HYPHEN_E_DOT]
    return requirements

setup(
    name="Churn Prediction",
    version="0.0.1",
    author="Akbarhusain",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
