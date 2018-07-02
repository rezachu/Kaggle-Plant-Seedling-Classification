# Kaggle Plant Seedlings Classification Kernel

## Overview
This is the kernel of Kaggle Competition - [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification)  where I trained my model on Google Cloud Platform and I constructed this kernel by using Keras with transferred Inception V3 pre-trained model.

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
- [iPython Notebook](http://ipython.org/notebook.html)
- [Keras](http://scikit-learn.org/stable/)
- CUDA 9.0 - `curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.88-1_amd64.deb`
- [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)
- Tensorflow-gpu - `pip install tensorflow-gpu`


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

## Kernel Structure
# 1. Data Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The number of category is 12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here are the training data intel:
-Black-grass 263 images
-Charlock 390 images
-Cleavers 287 images
-Common Chickweed 611 images
-Common wheat 221 images
-Fat Hen 475 images
-Loose Silky-bent 654 images
-Maize 221 images
-Scentless Mayweed 516 images
-Shepherds Purse 231 images
-Small-flowered Cranesbill 496 images
-Sugar beet 385 images

The Data is slightly imbalance that the minimum classes (Common wheat, Maize) have 221 images only and the maximum class (Loose Silky-bent) has 654 images.

# 2. Create Training and Testing Data Frame and Shuffle Data

Example Format:

![Image Show](https://github.com/rezachu/Kaggle-Plant-Seedling-Classification/blob/master/sources/df_preview.png)

Example Image:
![Image Show](https://github.com/rezachu/Kaggle-Plant-Seedling-Classification/blob/master/sources/example_01.png)


# 3. Read Data and Preprocess the Data by Removing the background

Training Data Size: 4750
Testing Data Size: 794
Train Valid Split: 0.2

- training_size: 3800
- valid_size: 950
- test_size: 794

Image resized to 299x299

Input Tensor Shape: (299, 299 ,3)

Example Image:
![Image Show](https://github.com/rezachu/Kaggle-Plant-Seedling-Classification/blob/master/sources/example_02.png)

# 4. Create FScore Function

```
from keras import backend as K

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fscore(y_true, y_pred):
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f_score = 2 * (p * r) / (p + r + K.epsilon())
    return f_score
```

# 5. Transfer Learning with Pre-trained Inception V3 Model

Model Summary:


```
model loading and compilation finished succesfully
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
inception_v3 (Model)         (None, 2048)              21802784  
_________________________________________________________________
dense_1 (Dense)              (None, 12)                24588     
=================================================================
Total params: 21,827,372
Trainable params: 21,792,940
Non-trainable params: 34,432
_________________________________________________________________

```

# 6. Model Training and Result


```
_________________________________________________________________

Hyparameter:
_________________________________________________________________

Batch Size: 64
EpochL 25
Optimizer: SGD
learning_rate: 25e-4
decay_rate: 0.0001 
momentum: 0.9
nesterovL True
Earlystop: Patience = 7
ReduceLROnPlateau: monitor='val_loss', factor=0.85, patience=3, min_lr=0.0001  

```

The best result was located at epoch 16 with 
- training_loss: 0.0085 
- training_accuracy: 0.9982
- training_fscore: 0.9987
- validating_loss: 0.1177
- validating_accuracy: 0.9642 
- validating_fscore:: 0.9651 

It took 3211.38s to training 22 epochs for 4750 image (train and valid) with batch size 64 on a single NVIDIA Tesla K80 GPU where each epoch took less then 140s. I trained it without GPU before which took around 6000s for an epoch.


```
Epoch 16/25
3800/3800 [==============================] - 138s 36ms/step - loss: 0.0085 - acc: 0.9982 - fscore: 0.9987 - val_loss: 0.1177 - val_acc: 0.9642 - val_fscore: 0.9651

.
.
.
.

Epoch 00022: val_acc did not improve from 0.96421
Epoch 23/25
3800/3800 [==============================] - 138s 36ms/step - loss: 0.0058 - acc: 0.9984 - fscore: 0.9987 - val_loss: 0.1098 - val_acc: 0.9611 - val_fscore: 0.9646

Epoch 00023: val_acc did not improve from 0.96421
3211.3844270706177
```

# 7. Predict the Test set and Create Submission.csv


# 8. Create Error Analysis

Run the `error_analysis.py` by only changing this line according to the file you would like to analysis

```
data = pd.read_csv("error_analysis_inceptionV3_masked_sgd_25e4lr_1e4dc_9e1f_30e_64b_.csv")
```

Example:
```
************************************************
Masked Result:
************************************************
Black-grass:  {'Positive': 29, 'Negative': 22} -- 56.86%
Charlock:  {'Positive': 70, 'Negative': 1} -- 98.59%
Cleavers:  {'Positive': 55, 'Negative': 0} -- 100.00%
Common Chickweed:  {'Positive': 115, 'Negative': 2} -- 98.29%
Common wheat:  {'Positive': 39, 'Negative': 4} -- 90.70%
Fat Hen:  {'Positive': 91, 'Negative': 7} -- 92.86%
Loose Silky-bent:  {'Positive': 136, 'Negative': 4} -- 97.14%
Maize:  {'Positive': 41, 'Negative': 3} -- 93.18%
Scentless Mayweed:  {'Positive': 111, 'Negative': 5} -- 95.69%
Shepherds Purse:  {'Positive': 41, 'Negative': 2} -- 95.35%
Small-flowered Cranesbill:  {'Positive': 95, 'Negative': 0} -- 100.00%
Sugar beet:  {'Positive': 75, 'Negative': 2} -- 97.40%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Black-grass
{'Black-grass': 29, 'Charlock': 0, 'Cleavers': 0, 'Common Chickweed': 0, 'Common wheat': 2, 'Fat Hen': 0, 'Loose Silky-bent': 19, 'Maize': 0, 'Scentless Mayweed': 0, 'Shepherds Purse': 0, 'Small-flowered Cranesbill': 0, 'Sugar beet': 1}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Charlock
{'Black-grass': 0, 'Charlock': 70, 'Cleavers': 0, 'Common Chickweed': 0, 'Common wheat': 0, 'Fat Hen': 1, 'Loose Silky-bent': 0, 'Maize': 0, 'Scentless Mayweed': 0, 'Shepherds Purse': 0, 'Small-flowered Cranesbill': 0, 'Sugar beet': 0}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cleavers
{'Black-grass': 0, 'Charlock': 0, 'Cleavers': 55, 'Common Chickweed': 0, 'Common wheat': 0, 'Fat Hen': 0, 'Loose Silky-bent': 0, 'Maize': 0, 'Scentless Mayweed': 0, 'Shepherds Purse': 0, 'Small-flowered Cranesbill': 0, 'Sugar beet': 0}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Common Chickweed
{'Black-grass': 0, 'Charlock': 0, 'Cleavers': 0, 'Common Chickweed': 115, 'Common wheat': 0, 'Fat Hen': 0, 'Loose Silky-bent': 0, 'Maize': 0, 'Scentless Mayweed': 1, 'Shepherds Purse': 1, 'Small-flowered Cranesbill': 0, 'Sugar beet': 0}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Common wheat
{'Black-grass': 0, 'Charlock': 0, 'Cleavers': 0, 'Common Chickweed': 0, 'Common wheat': 39, 'Fat Hen': 0, 'Loose Silky-bent': 2, 'Maize': 0, 'Scentless Mayweed': 1, 'Shepherds Purse': 0, 'Small-flowered Cranesbill': 0, 'Sugar beet': 1}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fat Hen
{'Black-grass': 0, 'Charlock': 0, 'Cleavers': 1, 'Common Chickweed': 1, 'Common wheat': 3, 'Fat Hen': 91, 'Loose Silky-bent': 1, 'Maize': 1, 'Scentless Mayweed': 0, 'Shepherds Purse': 0, 'Small-flowered Cranesbill': 0, 'Sugar beet': 0}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Loose Silky-bent
{'Black-grass': 3, 'Charlock': 0, 'Cleavers': 0, 'Common Chickweed': 0, 'Common wheat': 0, 'Fat Hen': 0, 'Loose Silky-bent': 136, 'Maize': 1, 'Scentless Mayweed': 0, 'Shepherds Purse': 0, 'Small-flowered Cranesbill': 0, 'Sugar beet': 0}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maize
{'Black-grass': 0, 'Charlock': 0, 'Cleavers': 0, 'Common Chickweed': 1, 'Common wheat': 0, 'Fat Hen': 0, 'Loose Silky-bent': 1, 'Maize': 41, 'Scentless Mayweed': 1, 'Shepherds Purse': 0, 'Small-flowered Cranesbill': 0, 'Sugar beet': 0}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Scentless Mayweed
{'Black-grass': 0, 'Charlock': 0, 'Cleavers': 1, 'Common Chickweed': 1, 'Common wheat': 0, 'Fat Hen': 0, 'Loose Silky-bent': 0, 'Maize': 0, 'Scentless Mayweed': 111, 'Shepherds Purse': 2, 'Small-flowered Cranesbill': 0, 'Sugar beet': 1}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shepherds Purse
{'Black-grass': 0, 'Charlock': 0, 'Cleavers': 0, 'Common Chickweed': 0, 'Common wheat': 0, 'Fat Hen': 0, 'Loose Silky-bent': 0, 'Maize': 0, 'Scentless Mayweed': 2, 'Shepherds Purse': 41, 'Small-flowered Cranesbill': 0, 'Sugar beet': 0}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Small-flowered Cranesbill
{'Black-grass': 0, 'Charlock': 0, 'Cleavers': 0, 'Common Chickweed': 0, 'Common wheat': 0, 'Fat Hen': 0, 'Loose Silky-bent': 0, 'Maize': 0, 'Scentless Mayweed': 0, 'Shepherds Purse': 0, 'Small-flowered Cranesbill': 95, 'Sugar beet': 0}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sugar beet
{'Black-grass': 0, 'Charlock': 0, 'Cleavers': 0, 'Common Chickweed': 1, 'Common wheat': 0, 'Fat Hen': 0, 'Loose Silky-bent': 1, 'Maize': 0, 'Scentless Mayweed': 0, 'Shepherds Purse': 0, 'Small-flowered Cranesbill': 0, 'Sugar beet': 75}

```






