from setuptools import setup

setup(name="pdecontrolgym",
        version="0.0.1", 
        install_requires=["gymnasium", 
            "numpy", 
            "matplotlib"], 
        )

# This is a setup script for a Python package named "pdecontrolgym".
# run pip install -e . terminal will find setup.py and install the package in editable mode.
# And load the codes at this document as a virtual environment.