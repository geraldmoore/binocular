from setuptools import setup, find_packages


setup(
    name="binocular",
    version="0.0.1",
    author="Gerald Moore",
    author_email="gerald.moore.mail@gmail.com",
    description="A Python package for grouping camera images based on similarity and capture time",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/your_repo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "pillow==10.3.0",
        "scikit-learn==1.5.0",
        "torch==2.3.0",
        "torchvision==0.18.0",
    ],
)
