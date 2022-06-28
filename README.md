<h1 align="center">pySelection for time series feature selection</h1> 

<p align="center"> 
<a href="https://github.com/lprtk/pySelection/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/lprtk/pySelection"></a> 
<a href="https://github.com/lprtk/pySelection/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/lprtk/pySelection"></a> 
<a href="https://github.com/lprtk/pySelection/stargazers"><img alt="Github Stars" src="https://img.shields.io/github/stars/lprtk/pySelection"></a> 
<a href="https://github.com/lprtk/pySelection/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/lprtk/pySelection"></a> 
<a href="https://github.com/lprtk/pySelection/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a> 
</p> 

## Table of contents 
* [Overview :loudspeaker:](#Overview)
* [Content :mag_right:](#Content)
* [Requirements :page_with_curl:](#Requirements)
* [File details :open_file_folder:](#File-details)
* [Features :computer:](#Features) 

<a id="section01"></a> 
## Overview 

<p align="justify">This is a small Python library for time series feature selection. The objective is to focus on building a "good" model. To define a "good" model, we rely on Theil's metrics (UM, US, UC, U1 or U) which allow us to conclude on the goodness of fit of the predictions made by a model. This library allows you to find the best model you need according to the criteria you want.<p> 

<a id="section02"></a> 
## Content 

For the moment, two class are available:
<ul> 
<li><p align="justify">The Metrics class implements all the Theil metrics as they are coded in the SAS software;</p></li> 
<li><p align="justify">The FeatureSelection class is the heart of the library. The fit function will allow you to estimate several combinations of sub-models and to calculate the scores (MAE, MSE, RMSE and Theil’s metrics). Then the other functions allow you to obtain the best model and its scores, all the models estimated during the iterations or to have a plot of the estimates in and out-of-sample.</p></li> 
</ul> 

<a id="section03"></a> 
## Requirements
* **Python version 3.9.7** 
* **Install requirements.txt** 
```console
$ pip install -r requirements.txt 
``` 

* **Librairies used**
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import choice, randint
from sklearn.linear_model import LinearRegression 
from sklearn.utils.validation import check_is_fitted 
``` 

<a id="section04"></a> 
## File details
* **requirements** 
* This folder contains a .txt file with all the packages and versions needed to run the project. 
* **pySelection** 
* This folder contains a .py file with all class, functions and methods. 
* **example** 
* This folder contains an example notebook to better understand how to use the different class and functions, and their outputs. 

</br> 

Here is the project pattern: 
```
- project 
    > pySelection
        > requirements 
            - requirements.txt
        > codefile 
            - pySelection.py
        > example 
            - pySelection.ipynb 
```

<a id="section05"></a> 
## Features 
<p align="center"><a href="https://github.com/lprtk/lprtk">My profil</a> • 
<a href="https://github.com/lprtk/lprtk">My GitHub</a></p> 
