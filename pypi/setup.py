from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ssakssree",
    version="0.1.2",
    author="Danda Company",
    author_email="datapod.k@gmail.com",
    description="A collection of statistics-based business analysis libraries by Danda Company, a professional analytics firm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dandacompany/ssakssree",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas>=1.0.0",
        "statsmodels>=0.12.0",
    ],
) 