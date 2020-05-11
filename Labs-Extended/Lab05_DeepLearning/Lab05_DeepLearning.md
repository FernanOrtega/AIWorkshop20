# Lab 05: Deep Learning

### Contents 

1. [Introduction](#introduction)
1. [Objective](#objective)
1. [Prerequisites](#prerequisites)
1. [Lab05 Description](#lab05-description)
1. [References](#references)

## Introduction
---

We have tested that Machine Learning algorithms are not bad at all.
We have developed a quick model that it is has an acceptable performance and we can use to classify all the things in the storage facility.
However, our friends suggested that how about if we could make a Deep Learning model for them.
They are thinking of making an app to classify your clothes and sell them automatically.
So, they are thinking that they need something powerful to develop this new adventure.

We think that a Deep Learning is good way to practice our new recent skills developed with Machine Learning and wanted to try it.
The reason to use Deep learning, and more precisely Convolutional Neural Networks (CNN) for image analysis, is that they perform well on complex data.
CNNs differ from traditional Artificial Neural Networks in the architectures that they have more neurons and layers, this make easier to learn features in complex data.
On the contrary and depending on the architecture it takes a lot to train them and they really need big amounts of data to generalize in a proper way.
It is a good practice to try to classify the equipment using a CNN, so let's do this!!

## Objective
---

In this Lab you will learn to face a Deep Learning approach to train a simply classifier.
We need that the classifier can distinguish between the different equipment in the storage facility.

## Prerequisites
---

* Have an Azure Machine Learning Workspace already deployed.
* Have a development environment ready.
* Install or update any Deep Learning framework you want to use.

## Lab05 Description
---

For the development of this Lab, you should use the notebook `Lab05_DeepLearning.ipynb`.
It is necessary to develop a classifier in any of the CNNs existing frameworks. 
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
Moreover, you should reserve a portion of the training dataset as a validation dataset. 
And of course, you are going to iterate on the architecture. 

For the training do not forget to register the metrics on the Azure Machine Learning workspace and make sure that the trained model is saved in the workspace.
In order to finish this lab you should get a model with a higher accuracy than the Machine Learning one and register the model in the Azure Machine Learning Workspace.

## References
---

* [Tensorflow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Pytorch](https://pytorch.org/)
* More about CNNs [here](https://cntk.ai/pythondocs/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.html#CNModel-Creation)
* [Splitting dataset ways](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)
* [Azure Machine Learning documentation](https://docs.microsoft.com/en-gb/azure/machine-learning/)
* [Azure Machine Learning SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py)