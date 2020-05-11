# Lab 04: Implementing Machine Learning models

### Contents 

1. [Introduction](#introduction)
1. [Objective](#objective)
1. [Prerequisites](#prerequisites)
1. [Lab03 Description](#lab03-description)
1. [References](#references)

## Introduction
---

This time we are going to build our own Machine Learning models. 
After seeing how the Custom Vision performs, it was easy to see that it was difficult sometimes to predict the actual class of a given image.
So, we suggested our friends that maybe we could create a Machine Learning model to adjust different parameters and see how this model classify the different images.

As we already know, working in machine learning is an iterative process and takes time to have fine tune model and the first step is usually to develop some baseline models to compare with. For that reason and for the sake of reducing time, *"There's no need to reinvent the wheel"*.
Thus, we are going to build our classifier using a common Machine Learning framework: [**Scikit-learn**](https://scikit-learn.org/stable/index.html)
This framework gives us the possibility to:
* Explore different algorithms quickly and see which one works best.
* Have a better control over the parameters of our classifier and its result to predict the right category.
* Lots of wisdom to help to configure the model.
* Know that the algorithms are correctly implemented and there are no bugs.
* Fast prototyping different classification models.

Futhermore, it may be cool to experiment with Deep Learning and compare a *classic* Machine Learning approach with one of this hyped models.

The reason to use Deep learning, and more precisely Convolutional Neural Networks (CNN) for image analysis, is that they perform well on complex data.
CNNs differ from traditional Artificial Neural Networks in the architectures that they have more neurons and layers, this make easier to learn features in complex data.
On the contrary and depending on the architecture it takes a lot to train them and they really need big amounts of data to generalize in a proper way.
It is a good practice to try to classify the equipment using a CNN, so let's do this!!

But how can we track different experiments and register models? 
As we are working with Azure, why not use an Azure Machine Learning Workspace to keep track on everything around the training of models and their registration?.
This kind of platform will help to keep track on the different algorithms developed for build the perfect classifier.
The reason to use them is to save all of our advance training and checking our hypothesis, so it is easier to obtain later the best model trained.

## Objective
---

In this Lab, we are going to create a Scikit-learn model, that tracks all the parameters and metrics in the Azure Machine Learning Workspace. In addiction, you will learn to face a Deep Learning approach to train a simply classifier.
After all the prototyping is done, we will register the best model among all the ones trained.

## Prerequisites
---

* Have an Azure Subscription ready to deploy an Azure Machine Learning Workspace.
* Have a development environment ready.
* Install or update any Deep Learning framework you want to use.

## Lab03 Description
---

For the development of this Lab use the notebook `Lab03_TrainingModels.ipynb`.
We are going to develop two image classifiers for our problem: a Random Forest classifier and a Convolutional Neural Network (Deep Learning) model. We are going to use the preprocessed images from Lab02.
As usual in machine learning, it is necessary to split the dataset into two sets: train and test.
Training dataset is used to train the model. 
Test dataset is used to test how the model classifies with unknown data samples.
A typical distribution of these datasets is 70% for train dataset and 30% of test dataset to ensure the model is generalizing in a proper way.

The Random Forest classifier is implemented by means of Scikit-learn. It is a good library to start with machine learning and it is largely used in production systems.

The second classifier will be implemented by means of Tensorflow with Keras, which is the most used framework for deep learning and provides an implementation of Convolutional Neural Networks (CNNs). 

The idea is to have a model with an architecture similar to:

1. Input Layer (3 channel image input layer)
2. Convolutional (2D)
3. Max Pooling
4. Convolutional (2D)
5. Max Pooling
6. Dense (Output layer)

This should be enough to create a CNN that can be used to classify the equipment.
In addition, if you want to train faster you can use GPU development environment making the appropriate adjustments.
Train a model on the training dataset using the suggested architecture or an equivalent that the you wanted to try. 

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