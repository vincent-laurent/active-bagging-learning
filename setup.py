from setuptools import setup, find_packages


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


install_reqs = parse_requirements("requirements.txt")

setup(
    name="Surrogate-Models",
    version='0.0.1',
    author="Vincent LAURENT",
    author_email="vlaurent@eurobios.com",
    long_description=open('README.md').read(),
    include_package_data=True,
    install_requires=install_reqs,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 0 - Planning",
        "Natural Language :: French",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Surrogate Modeling",
    ],
)
