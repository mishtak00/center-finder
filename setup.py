from setuptools import setup, find_packages
from setuptools.command.install import install

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == '__main__':
    setup(
        name="centerfinder",
        version="0.0.1",
        author="Yujie Liu, Gebri Mishtaku",
        author_email="yliu134@u.rochester.edu, gmishtak@u.rochester.edu",
        description="Python implementation of the center-finding algorithm",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/mishtak00/center-finder",
        packages=find_packages(exclude=["run", "data", "output"]),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
    )

