# oracle crocubot [![CircleCI](https://circleci.com/gh/alpha-i/oracle-crocubot-python.svg?style=svg&circle-token=f6a7198d3b32ae0fb56dfec1daee167a930445eb)](https://circleci.com/gh/alpha-i/oracle-crocubot-python)


A library for predicting returns using Bayes by Backprop


![Crocubot](docs/crocubot.jpg "Crocubot")

## Setup Development Environment

### Create conda environment
```bash
$ conda create -n crocubot-env python=3.5
$ source activate crocubot-env
```

### Install dependencies

```bash
$ pip install -U setuptools --ignore-installed --no-cache-dir
$ pip install -r requirements.txt --src $CONDA_PREFIX
$ pip install -r requirements_alphai.txt --src $CONDA_PREFIX
$ pip install -r requirements_dev.txt
```

### Running the test suite
```bash
$ PYTHONPATH=. python -m pytest tests/
```


## Installation on Windows
We need to install `scipy` and `scikit-learn` using `conda` as it requires some external libraries which are not easy to build in Windows.
```commandline
conda install numpy scipy pandas pytables scikit-learn statsmodels cython
```
We will need to set the variables for the GSL in the environment first. 
```commandline
set GSL_INCLUDE_DIR=C:\Users\sree\Documents\Software\gsl\include
set GSL_LIBRARY_DIR=C:\Users\sree\Documents\Software\gsl\lib
```
Now install the requirements using `pip`.
```commandline
pip install -r requirements.txt --src %CONDA_PREFIX%
pip install -r requirements_alphai.txt --src %CONDA_PREFIX%
pip install -r requirements_dev.txt
```

### Known Issues
There is an issue with the installation of `pytables`. 
The `pip` installation fails with `pip` not being able to locate `hdf5` library. This can be avoided by setting the 
variable `HDF5_DIR` to the directory of anaconda installation. Using the bash this can be done by
```bash
export HDF5_DIR=path-to-anaconda
```
where `path-to-anaconda` is the full path to `anaconda` installation (for example `export HDF5_DIR=$HOME/anaconda3`)

After doing this, it may be necessary to install `pytables` using conda, which will take care of the necessay hdf5 
libraries, and then update it to the correct version using `pip`:
```bash
 conda install pytables
 pip install tables --src $CONDA_PREFIX --ignore-installed --no-cache-dir
```

Another related issue arises when running the `pytest`. A runtime error occurs because the `lhdf5` is not visible to 
the executable. This  can be solved by setting the `DYLD_LIBRARY_PATH` to `lib` directory of anaconda installation.
Again, using bash
```bash
export DYLD_LIBRARY_PATH=path-to-anaconda/lib
```


