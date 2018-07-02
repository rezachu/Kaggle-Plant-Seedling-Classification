# Kaggle Plant Seedlings Classification Kernel

## Overview
This is the kernel of Kaggle Competition - [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification) and I constructed this kernel by using Keras and transferred three pre-trained models - VGG 16, Resnet 50 and Inception V3. And I trained my model on Google Cloud Platform - 

## Compute Engine Instance Configuration
- 4 CPUS (26 GB memory)
- 1 NVIDIA Tesla K80 GPU 
- 150 GB bootdisk 
- Ubuntu 16.04 LTS environment.

## Software requirement
- [Python 3.5](https://www.python.org/downloads/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [Keras](http://scikit-learn.org/stable/)
- CUDA 9.0 - `curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.88-1_amd64.deb`
- [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)
- Tensorflow-gpu - `pip install tensorflow-gpu`
- [iPython Notebook](http://ipython.org/notebook.html)

## Documents included in the repository
- `README.md`
- `Plant_Classifier_Keras_tuning_InceptionV3.ipynb`
- `error_analysis.py`

### Memo to Google Cloud Platform beginner
In case you are new to Google Cloud Platform (GCP) and you do not know how to create a GCP bucket and instance, my Medium blog post is here to help:
- [A Complete Step by Step Guide of Keras Transfer Learning with GPU on Google Cloud Platform](https://medium.com/datadriveninvestor/complete-step-by-step-guide-of-keras-transfer-learning-with-gpu-on-google-cloud-platform-ed21e33e0b1d)

## Run 
1. Start your GCP instance
2. Start the SSH terminal
3. Activate your environment
4. Run `jupyter-notebook --no-browser --port=<port_number>`
5. Type this url in your browser `http://<external_IP_address_of_instance>:<port_number>/`

## Data
We would us the data provide by [Kaggle](https://www.kaggle.com/c/plant-seedlings-classification/data) and there are 3 folders.

1. train.zip
2. test.zip
3. sample_submission.csv

There are twevle classes in the train data directory
1. Black-grass
2. Charlock
3. Cleavers
4. Common Chickweed
5. Common wheat
6. Fat Hen
7. Loose Silky-bent
8. Maize
9. Scentless Mayweed
10. Shepherds Purse
11. Small-flowered Cranesbill
12. Sugar beet

