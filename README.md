# Traffic Signs Detection using CNN
 

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify German traffic signs using the GTSRB (German Traffic Sign Recognition Benchmark) dataset. The dataset can be found [here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Conclusion](#conclusion)

## Introduction

This project uses a CNN model to classify images of German traffic signs. The CNN model is trained and evaluated on the GTSRB dataset, which contains images of 43 different types of traffic signs.

## Dataset

The dataset used in this project is the GTSRB dataset. It consists of training images and test images of German traffic signs. The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

## Prerequisites

To run this project, you need to have the following libraries installed:

- NumPy
- Pandas
- Matplotlib
- OpenCV
- TensorFlow
- Keras
- scikit-learn

## Installation

To install the required libraries, you can use the following commands:

```bash
pip install numpy pandas matplotlib opencv-python tensorflow keras scikit-learn
```

## Usage

1. Download the GTSRB dataset from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
2. Extract the dataset to a directory of your choice.
3. Update the `cur_path` variable in the code to point to the directory where the dataset is located.

## Model Architecture

The CNN model used in this project consists of the following layers:

1. Conv2D layer with 32 filters and (5, 5) kernel size, followed by ReLU activation.
2. Conv2D layer with 32 filters and (5, 5) kernel size, followed by ReLU activation.
3. MaxPool2D layer with (2, 2) pool size.
4. Dropout layer with 0.25 dropout rate.
5. Conv2D layer with 64 filters and (3, 3) kernel size, followed by ReLU activation.
6. Conv2D layer with 64 filters and (3, 3) kernel size, followed by ReLU activation.
7. MaxPool2D layer with (2, 2) pool size.
8. Dropout layer with 0.25 dropout rate.
9. Flatten layer.
10. Dense layer with 256 units and ReLU activation.
11. Dropout layer with 0.5 dropout rate.
12. Dense layer with 43 units and softmax activation.

## Training

The model is trained using the training images from the GTSRB dataset. The training process involves the following steps:

1. Loading and preprocessing the images.
2. Splitting the data into training and validation sets.
3. Converting the labels to categorical format.
4. Defining the model architecture.
5. Compiling the model with categorical cross-entropy loss and Adam optimizer.
6. Training the model for a specified number of epochs.

## Evaluation

The model's performance is evaluated using the test images from the GTSRB dataset. The evaluation process involves the following steps:

1. Loading and preprocessing the test images.
2. Evaluating the model on the test images.
3. Calculating the accuracy score.
4. Generating a classification report.

## Prediction

To make predictions on new images, use the `test_on_img` function. This function loads and preprocesses an image, and then uses the trained model to predict the class of the traffic sign in the image.



## Conclusion

This project demonstrates how to build and train a CNN model to classify German traffic signs using the GTSRB dataset. The model achieves high accuracy on both the training and test sets. The code can be easily extended and adapted to other image classification tasks.
