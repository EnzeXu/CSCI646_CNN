Assignment 1 - CNNs
===

# 1. Introduction
In this assignment, please use Pytorch (not Tensorflow) to implement the convolutional neural networks (CNNs) on the CIFAR 10 dataset (60,000 small square 3x32×32 pixel color images ( 3 input channels for color image)). You can directly use the data loader from Pytorch to load the training/test data samples. Please see the data loader [example in this link](https://www.notion.so/Assignment-1-CNNs-Work-Individually-fc70ab851ccd4c039c7780239b3a7f10?pvs=4#031daf3ef9474edd85a802a5ded6b84c).  About detailed CIFAR10 dataset introductions, please [refer to this website](https://www.notion.so/Assignment-1-CNNs-Work-Individually-fc70ab851ccd4c039c7780239b3a7f10?pvs=4#031daf3ef9474edd85a802a5ded6b84c)

# 2. Getting Started

This project is developed using Python 3.9+ and is compatible with macOS, Linux, and Windows operating systems.

## 2.1 Preparations

(1) Clone the repository to your workspace.

```shell
~ $ git clone https://github.com/EnzeXu/CSCI646_CNN.git
```

(2) Navigate into the repository.
```shell
~ $ cd CSCI646_CNN
~/CSCI646_CNN $
```

(3) Create a new virtual environment and activate it. In this case we use Virtualenv environment (Here we assume you have installed the `virtualenv` package using you source python script), you can use other virtual environments instead (like conda).

For macOS or Linux operating systems:
```shell
~/CSCI646_CNN $ python -m venv ./venv/
~/CSCI646_CNN $ source venv/bin/activate
(venv) ~/CSCI646_CNN $ 
```

For Windows operating systems:

```shell
~/CSCI646_CNN $ python -m venv ./venv/
~/CSCI646_CNN $ .\venv\Scripts\activate
(venv) ~/CSCI646_CNN $ 
```

You can use the command deactivate to exit the virtual environment at any time.

## 2.3 Install Packages

Install dependent packages.
```shell
(venv) ~/CSCI646_CNN $ pip install -r requirements.txt
```


## 2.4 Download Datasets

The datasets CIFAR10 and CIFAR100 will be downloaded once automatically during the training.


## 2.5 Run Training

(1) Run Training on CIFAR10 using CNNs:

```shell
(venv) ~/CSCI646_CNN $ python run.py --prob CIFAR10-CNN --gpu id 0 --seed 0
```

The accuracy obtained on the test set would be around 79.49%.

(2) Run Training on CIFAR100 using ResNet18:.
```shell
(venv) ~/CSCI646_CNN $ python run.py --prob CIFAR100-ResNet18 --gpu id 0 --seed 0
```

The accuracy obtained on the test set would be around 73.17%.


# 3. Questions

If you have any questions, please contact xezpku@gmail.com.