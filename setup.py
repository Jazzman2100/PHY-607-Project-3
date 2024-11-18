#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="fm_mcmc",  # Name of your package
    version="0.1.0",  # Initial version
    description="Implementing Markov Chain Monte Carlo simulation to estimate parameters of FM signals",  # Fix the description string
    long_description=open("README.md").read(),  # Optional: You can write a README file
    long_description_content_type="text/markdown",  # If your README is in Markdown
    author="Jasmine, Nich",  # Replace with your name or the author of the package
    author_email="yhuan223@syr.edu, nmrubayi@syr.edu",  # Replace with your email addresses
    url="https://github.com/Jazzman2100/project1.git",  # Replace with the actual URL (if applicable)
    packages=find_packages(),  # Automatically find all submodules (i.e., fm_mcmc package)
    install_requires=[
        "numpy",      # Required dependency
        "matplotlib", # Required for plotting
        "emcee",      # Required for MCMC sampling
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license type
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version supported
    entry_points={
        'console_scripts': [
            'fm-mcmc=main:main',  # Ensure this is pointing to the main function in main.py (at root)
        ],
    },
)
