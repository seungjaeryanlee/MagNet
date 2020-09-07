import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mag-net",
    version="0.0.1",
    author="Seungjae Ryan Lee",
    author_email="seungjaeryanlee@gmail.com",
    description="MagNet is a large-scale dataset designed to enable researchers modeling magnetic core loss using machine learning to accelerate the design process of power electronics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seungjaeryanlee/MagNet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
