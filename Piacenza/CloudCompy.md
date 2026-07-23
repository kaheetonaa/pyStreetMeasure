## Features

The installation instructions are in the package, in doc/UseLinuxCondaBinary.md

This version does not require Conda anymore but requires a Python 3.12 environment with at least the following packages: 
numpy requests psutil scipy numpy-quaternion cmake matplotlib
 
For instance, create a Python virtual environment with venv (adapt the path for Your Python 3.12 install and for the virtual env). In the terminal :
path/to/python3.12 -m venv ${HOME}/.venv312
source ${HOME}/.venv312/bin/activate
python3 -m pip install --upgrade pip
pip install numpy scipy requests psutil matplotlib numpy-quaternion cmake
 
Unzip the CloudcomPy binary in the directory of your choice.
 
Before using CloudCompare or CloudComPy, you need to load the environment, with 2 steps :
 - The Python virtual environment:
source ${HOME}/.venv312/bin/activate
 - The paths (PYTHONPATH, PATH) required for cloudComPy and CloudCompare:
cd path/to/CloudComPy312
source bin/envCloudComPy.sh activate
 
The documentaion is provided with the package. From the Python prompt : 
import cloudComPy as cc
cc.launchDoc()

