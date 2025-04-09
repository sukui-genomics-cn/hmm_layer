from setuptools import setup, find_packages

setup(
    name="hmm-layer",
    version="0.2.0",
    description="A PyTorch-based Hidden Markov Model (HMM) Layer for Genomic Structure Prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/sukui-genomics-cn/hmm_layer.git",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scipy",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="HMM, Hidden Markov Model, PyTorch, Genomic Prediction, Bioinformatics",
)