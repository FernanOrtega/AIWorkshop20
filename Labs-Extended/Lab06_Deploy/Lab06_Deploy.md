# Lab 06: Deploy a web service, the last model

### Contents 

1. [Introduction](#introduction)
1. [Objective](#objective)
1. [Prerequisites](#prerequisites)
1. [Lab06 Description](#lab06-description)
1. [References](#references)

## Introduction
---

We made it!!!
Finally, we have developed a custom model that can distinguish the piece of equipment among all the gear that our friends have.
However, our friends said they need to consume this models somehow.
They have uploaded all their photos to a cloud storage and they wanted to pass the url to the service in order to try it.
We are thinking on deploying it as a webservice so our friends can send all the information they need and know which category is the equipment they send.

In any machine learning project, it is not just enough to create a model that solve your problem.
You also need to know how to deploy it so anyone apart from you can consume it and provide with the functionality you have been requested.
There are different ways to implement this, you can use cloud services specific for this tasks as Azure or you can try to build your own API an expose it to the rest.
However, this last approach it is more elaborated and take time to develop an API and put through a web service.

Nowadays, thanks to the built-in cloud services it is a lot more easier, you only need to provide some pieces.
The rest will be done transparent to you in order to create the webservice.
A standard use in this way is to containerized the model and productionalize through any service that support the deployment of containers.
In this case we are going to use Azure Machine Learning Workspace in order to create a Webservice of the model.

## Objective
---

The goal of this final lab is to deploy the developed CNN model into a webservice in which we can pass an image url and give back the category of the equipment.
To accomplish this task we are going to use Azure Machine Learning Workspace.

## Prerequisites
---

* Have a development environment.
* Have access to an Azure Machine Learning Workspace.
* Have a trained CNN model.

## Lab06 Description
---

For the development of this Lab, you should use the notebook `Lab06_Deploy.ipynb`.
The idea of this Lab is to take the resultant trained model of *Lab05* and deploy it into a web service.
You should the registered model in the workspace to be deployed.
The input of the webservice should be the image url of the equipment and the output of the webservice will be category predicted by the model.

For the deployment you will need to create a python script:
* Have an initialization method of the model.
* Data preprocessing step from the image to scale and normalize it.
* Have a run method to predict the category.

An example of this scrip can be seen in [here](https://docs.microsoft.com/es-es/azure/machine-learning/service/tutorial-deploy-models-with-aml#create-scoring-script)
In addition you should create also a conda environment file to install the different packages.
You should use the Azure ML SDK to create the docker image with the model, score script and conda environment. 
Finally, after creating the image, deploy it to any container service in azure to test it.

## References
---

* [Azure Machine Learning documentation](https://docs.microsoft.com/en-gb/azure/machine-learning/)
* [Azure Machine Learning SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py)
* [Model deployment](https://docs.microsoft.com/es-es/azure/machine-learning/service/how-to-deploy-and-where)
* [Docker](https://docs.docker.com/get-started/)