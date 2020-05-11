# Lab 03: A data science job

### Contents 

1. [Introduction](#introduction)
1. [Objective](#objective)
1. [Prerequisites](#prerequisites)
1. [Lab03 Description](#lab03-description)
1. [References](#references)

## Introduction
---

After doing some research with the *"out of the box"* models and services from Azure, we have check that it can be done quickly but they have some problems.
In addition, they don't solve completely our problem and we cannot tune them to have a better response.
As we cannot customized them, we need to implement our own classifier that meets all our necessities.
And as in any worthy data science project, we will start from small to big. 

First we need to work on the data.
We will need a dataset that can be used to train any model.
For this reason, we need to preprocess the images to make them consistent and comparable.

## Objective
---

After finishing this *Lab*, you will obtain an appropriate dataset, with normalized and comparable images in order to train any *Machine Learning* model later on.

## Prerequisites
---
* Have a development environment ready.

## Lab03 Description
---

Our friends provided us with an image dataset of different equipment.
This information consists of raw and unformatted pixel data.
Before training any Machine Learning model, there is a task called **Data Preparation** or **Data Preprocessing**.
This task consists of cleaning and preprocessing the data in order to have a normalized and balanced dataset.
It is important to emphasize that without the appropriate data, we can already train any model that won't fulfill the task for which it was designed.
*Data preprocessing* is a crucial task and much of the time goes into this step to design a good input data to ours Machine Learning models.

Depending on the data the steps needed are different. 
In the case of working with images, the data to treat are the value of the pixel and some of the operations are: contrast enhancement, equalization, rotation, translation, etc.
Some of the problems we can face are that the pictures have glares, saturation differences, contrast issues, etc. that make difficult to compare them or extract some features from them with a Machine Learning model.

The processed dataset will consist of the same set of categories and images but normalized and equalized with a dimension of 128x128.
But the image will maintain the same aspect ratio.
Thus, the *data preprocessing* will consist of:
* Pad the images, that do not have a square aspect ratio.
* Reshape the image to 128 pixels width, 128 pixels height and 3 channels of color (do not stretch them).
* Equalize the pixel intensity within the range 0 to 255. 
* Save the processed data to use it later.

For the development of this preprocessing step, you should use the `Lab03_DataPreparation.ipynb`.

## References
---

* [Glob module](https://docs.python.org/3/library/glob.html#module-glob)
* [Image processing with Numpy](https://www.degeneratestate.org/posts/2016/Oct/23/image-processing-with-numpy/)
* [Matplotlib library](https://matplotlib.org/)
* [Pillow Image module](https://pillow.readthedocs.io/en/4.2.x/reference/Image.html)
* [Pillow Image Operations module](http://pillow.readthedocs.io/en/4.2.x/reference/ImageOps.html) 