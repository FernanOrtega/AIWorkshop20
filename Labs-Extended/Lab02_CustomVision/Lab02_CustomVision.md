# Lab 02: Something more powerful

### Contents 

1. [Introduction](#introduction)
1. [Objective](#objective)
1. [Prerequisites](#prerequisites)
1. [Lab02 Description](#lab02-description)
1. [References](#references)

## Introduction
---

After seeing what cognitive services are capable of, we realizes that they respond pretty well with general purpose objects (shoes, hats, etc.).
However, some specific objects as ropes, they are not well aware of them and they are not capable to detect them.
We want to create a custom classifier that can recognize the product from the image requested.
There is another Microsoft service that we want to test, it is the [Custom Vision service](https://azure.microsoft.com/es-es/services/cognitive-services/custom-vision-service/) to create this custom model.

Custom Vision is a cognitive service that leads you build, deploy, and improve your own image classifier.
This image classifier consists of applying different labels or classes to the images according to their visual characteristics.
The idea is that you can upload the set of images corresponding to a class.
After defining all the classes and upload your training dataset, just train your custom model.
So, the service make a transfer learning training for your model and you will obtain a model that will classify the images depending on the categories you have specified. 

In order to test this service, for a quick test we want to classify the different objects that our image dataset had.
We are going to create a web service to request which element belongs to the different categories in the dataset.
And we are going to use different images that the ones from the dataset in order to see how this model generalizes.

## Objective
---

The goal of this lab is to build and test a Custom Vision model created with the data of the image dataset given.
To fulfill this Lab you must call the classification endpoint of your model and predict successfully the class of object that you are passing to it.

## Prerequisites
---

* Have a development environment ready.
* Have an Azure Cognitive Services already deployed in your subscription.
* Have access to Custom Vision portal from [here](https://customvision.ai)
        
## Lab02 Description
---

You should use the notebook `Lab02_CustomVision.ipynb` to create a classification model to predict whether an image corresponds to any of the 12 categories of the dataset.
We recommend to use a small amount of images (between 20-30) so you can create them more quickly.

Once you have create the model with the different tags trained it and publish the model once you are satisfied with your results.
When the model has been published, call the prediction endpoint using python and predict the class of an image not used in the training.
This can be an image from dataset that was not uploaded or an image found online.

> *Note that creating the project, creating the tags, training the model and publishing it can be done through the web interface from [here](https://www.customvision.ai/)*

## References
---
* [Custom Vision Docs](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/)
* [Custom Vision SDK](https://docs.microsoft.com/en-gb/python/api/overview/azure/cognitiveservices/customvision?view=azure-python)
* [Quickstart: How to build a classifier with Custom Vision](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/getting-started-build-a-classifier)