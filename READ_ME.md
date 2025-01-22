# Classifier Model Optimization: Tailoring Machine Learning Models to COVID patients classification
***
## Introduction
This project aims to evaluate the accuracy of different machine learning algorithms on the task of determining if COVID patients
should be hospitalized or not.

The analysis focuses on choosing the best hyperparameters for each model, using the optuna framework, and comparing each
models' accuracies.
***
## Configuration
Customize the experiment using the **config.ini** file. 
Key parameters include:
### **Types**:
- **data_type**: Choose `img_feature` for numerics and images features vectors combined, `img` for only raw images, and `numeric` for only the numeric data.
- **model_type**: Specify `dt`, `nn`, `cnn` or `dl` for the model type.
***
## Running
After the environment for the experiment is built, simply run **main.py** and check the results on the terminal
***
## Analysing Data
Each test will compare different hyperparameters and their accuracies. At the end, the best combination of hyperparameters
will be printed on the terminal, with the according result.
## Framework
This project was built mainly using the Optuna framework, more information about the intricacies of its functioning
can be found [here](https://optuna.readthedocs.io/en/stable/). 
***
## Disclaimer
This repository is an addition to an academic paper 
with the same name, for the **Machine Learning** class at **University of Coimbra**, Portugal.

The authors are:
* Catarina Silva
* Mariana Guiomar 
* Saulo José Mendes
