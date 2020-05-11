# Lab 04: Machine Learning 101

### Contents 

1. [Introduction](#introduction)
1. [Objective](#objective)
1. [Prerequisites](#prerequisites)
1. [Lab04 Description](#lab04-description)
1. [References](#references)

## Introduction
---

This time we are going to build our own Machine Learning model. 
After seeing how the Custom Vision performs, it was easy to see that it was difficult sometimes to predict the real class of the image given.
So, we suggested our friends that maybe we could create a Machine Learning model to adjust different parameters and see how this model classify the different images.

As we already know, working in machine learning is an iterative process and takes time to have fine tune model.
For that reason, the first thing we have to do is to test some hypothesis.  
It is in this moment, when we remember the first rule of the machine learning developer: *"There's no need to reinvent the wheel"*.
Thus, we are going to build our classifier using a common Machine Learning framework: [**Scikit-learn**](https://scikit-learn.org/stable/index.html)
This framework gives us the possibility to:
* Explore different algorithms quickly and see which one works best.
* Have a better control over the parameters of our classifier and its result to predict the right category.
* Lots of wisdom to help to configure the model.
* Know that the algorithms are correctly implemented and there are no bugs.
* Fast prototyping different classification models.

But how can we track different experiments and register models? 
Well as we are working with Azure how about if use the Azure Machine Learning Workspace to keep track on everything around the training of models and their registration.
This kind of platform will help to keep track on the different algorithms developed for build the perfect classifier.
The reason to use them is to save all of our advance training and checking our hypothesis, so it is easier to obtain later the best model trained.

## Objective
---

In this Lab, we are going to create a Scikit-learn model, that tracks all the parameters and metrics in the Azure Machine Learning Workspace.
After all the prototyping is done, we will register the best model among all the ones trained.

## Prerequisites
---

* Have an Azure Subscription ready to deploy an Azure Machine Learning Workspace.
* Have a development environment ready

## Lab04 Description
---

For the development of this Lab use the notebook `Lab04_MachineLearning101.ipynb`.
We are going to develop an image classifier for our problem using one the classification methods that are inside Scikit-learn. 
Scikit-learn is a common used library for starting out and is not uncommon in production systems.
It is a very good framework to start learning, implementing and programming the most common Machine Learning algorithms.

In the Lab you will develop a classification model to predict the class of each of the different classes given in the dataset.
For this task used the preprocessed images done in Lab03.
The algorithm that you chose could any of the traditional non-parametric algorithms (e.g. random forest, support vector machines, etc.).

As in any task of training of machine learning it is necessary to split the dataset in train and test data.
The training dataset is used to train the model and they are unique data used to teach the model how to behave.
Whereas the test data it will be never showed during the training to the model, so it is really useful to validate that our model behaves in the way we wanted.
Normally, these datasets are splitted in a 70/30 fashion due to have a way to validate the model is generalizing in a proper way.

In addition, we are going to use another tool from Azure, Azure Machine Learning Workspace since we want to store all the information from the trainings.
You will need to create a workspace in your subscription and then create an experiment to log all the training information inside.
Finally, you must register the best model found during the training.

To fulfill this Lab you need to:
* Create an Azure Machine Learning Workspace
* Create an experiment
* Load the data and split it in train and test datasets.
* Select a classification method from Scikit-Learn
* Uses some metrics to compare each training: accuracy, precision, recall, f1-score and confusion matrix.
* Build the training
* Log all the important training information in Az ML.
* Register the best model in order to save it.

## References
---

* [Scikit-learn](https://scikit-learn.org/stable/index.html)
* [Splitting dataset ways](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)
* [Azure Machine Learning documentation](https://docs.microsoft.com/en-gb/azure/machine-learning/)
* [Azure Machine Learning SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py)