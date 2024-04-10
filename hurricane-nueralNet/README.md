# Hurrican Damage Prediction with Nueral Networks

This repository focuses on classifying satellite images from Texas after Hurricane Harvey into "damaged" and "non-damaged" buildings using TensorFlow nueral networks. We explored various neural network architectures, deploying the best-performing model with a Flask-based inference server. This README covers the deployment and usage of the model inference server.

## Model Training
### Data:
The data that was used for this project can be found in the `/data` directory of this repository. It contains 21322 (128px, 128px) RGB images. Each image was classified as "damage" or "no_damage". The RGB values were normalized to the range [0, 1] for model training. After that, the data was split for testing and training and several different nueral network architectures were used.

### Architectures:

We experimented with several neural network architectures, including:

- **ANN (Artificial Neural Network)**: A dense, fully-connected network with varying layers and perceptrons.

```_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten_1 (Flatten)         (None, 49152)             0         
                                                                 
 dense_8 (Dense)             (None, 1028)              50529284  
                                                                 
 dense_9 (Dense)             (None, 512)               526848    
                                                                 
 dense_10 (Dense)            (None, 256)               131328    
                                                                 
 dense_11 (Dense)            (None, 128)               32896     
                                                                 
 dense_12 (Dense)            (None, 128)               16512     
                                                                 
 dense_13 (Dense)            (None, 64)                8256      
                                                                 
 dense_14 (Dense)            (None, 32)                2080      
                                                                 
 dense_15 (Dense)            (None, 1)                 33        
                                                                 
=================================================================
Total params: 51247237 (195.49 MB)
Trainable params: 51247237 (195.49 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

- **LeNet-5 CNN**: A convolutional neural network based on the classical Lenet-5 architecture, adjusted for our image dimensions.

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_4 (Conv2D)           (None, 124, 124, 6)       456       
                                                                 
 average_pooling2d (Average  (None, 62, 62, 6)         0         
 Pooling2D)                                                      
                                                                 
 conv2d_5 (Conv2D)           (None, 58, 58, 16)        2416      
                                                                 
 average_pooling2d_1 (Avera  (None, 29, 29, 16)        0         
 gePooling2D)                                                    
                                                                 
 flatten_3 (Flatten)         (None, 13456)             0         
                                                                 
 dense_24 (Dense)            (None, 120)               1614840   
                                                                 
 dense_25 (Dense)            (None, 84)                10164     
                                                                 
 dense_26 (Dense)            (None, 1)                 85        
                                                                 
=================================================================
Total params: 1627961 (6.21 MB)
Trainable params: 1627961 (6.21 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

- **Alternate-LeNet-5 CNN**: Inspired by a variant discussed in [this research paper](https://arxiv.org/pdf/1807.01688.pdf), though customized for our specific dataset.

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_950 (Conv2D)         (None, 126, 126, 32)      896       
                                                                 
 max_pooling2d_44 (MaxPooli  (None, 63, 63, 32)        0         
 ng2D)                                                           
                                                                 
 conv2d_951 (Conv2D)         (None, 61, 61, 64)        18496     
                                                                 
 max_pooling2d_45 (MaxPooli  (None, 30, 30, 64)        0         
 ng2D)                                                           
                                                                 
 conv2d_952 (Conv2D)         (None, 28, 28, 128)       73856     
                                                                 
 max_pooling2d_46 (MaxPooli  (None, 14, 14, 128)       0         
 ng2D)                                                           
                                                                 
 conv2d_953 (Conv2D)         (None, 12, 12, 128)       147584    
                                                                 
 max_pooling2d_47 (MaxPooli  (None, 6, 6, 128)         0         
 ng2D)                                                           
                                                                 
 flatten_6 (Flatten)         (None, 4608)              0         
                                                                 
 dropout_13 (Dropout)        (None, 4608)              0         
                                                                 
 dense_62 (Dense)            (None, 512)               2359808   
                                                                 
 dense_63 (Dense)            (None, 1)                 513       
                                                                 
=================================================================
Total params: 2601153 (9.92 MB)
Trainable params: 2601153 (9.92 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
- VGG: A custon model with the VGG-16 architecture as described by [this paper](https://arxiv.org/pdf/1409.1556.pdf)
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 vgg16 (Functional)          (None, 4, 4, 512)         14714688  
                                                                 
 flatten_5 (Flatten)         (None, 8192)              0         
                                                                 
 dense_53 (Dense)            (None, 512)               4194816   
                                                                 
 dense_54 (Dense)            (None, 256)               131328    
                                                                 
 dropout_9 (Dropout)         (None, 256)               0         
                                                                 
 dense_55 (Dense)            (None, 128)               32896     
                                                                 
 dense_56 (Dense)            (None, 32)                4128      
                                                                 
 dropout_10 (Dropout)        (None, 32)                0         
                                                                 
 dense_57 (Dense)            (None, 1)                 33        
                                                                 
=================================================================
Total params: 19077889 (72.78 MB)
Trainable params: 4363201 (16.64 MB)
Non-trainable params: 14714688 (56.13 MB)
_________________________________________________________________
```
- ResNet: A custon model with the ResNet architecture as described by [this paper](https://arxiv.org/pdf/1512.03385.pdf)
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 resnet50 (Functional)       (None, 4, 4, 2048)        23587712  
                                                                 
 flatten (Flatten)           (None, 32768)             0         
                                                                 
 dense (Dense)               (None, 256)               8388864   
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 64)                8256      
                                                                 
 dense_3 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 32017793 (122.14 MB)
Trainable params: 8430081 (32.16 MB)
Non-trainable params: 23587712 (89.98 MB)
_________________________________________________________________
```

- Xception: A custon model with the Xception architecture from Google as described by [this paper](https://arxiv.org/pdf/1610.02357.pdf)
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 resizing (Resizing)         (None, 299, 299, 3)       0         
                                                                 
 xception (Functional)       (None, 1000)              22910480  
                                                                 
 dense (Dense)               (None, 128)               128128    
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               16512     
                                                                 
 dense_2 (Dense)             (None, 1)                 129       
                                                                 
=================================================================
Total params: 23055249 (87.95 MB)
Trainable params: 144769 (565.50 KB)
Non-trainable params: 22910480 (87.40 MB)
_________________________________________________________________
```


### Evaluation and Model Selection
Models were evaluated based on their accuracy score on the validation set. 

The `alt_lenet` model was the best model.


# TensorFlow Model Serving API

## Overview

This API serves as an interface for classifying images using pre-trained TensorFlow models. It allows users to classify images as "damaged" or "not damaged", change the underlying model, and retrieve information about the current model.

## Use

### API Endpoints

| Endpoint            | Method | Description                                                                                         |
|---------------------|--------|-----------------------------------------------------------------------------------------------------|
| `/model/info`       | GET    | Provides basic information about the currently loaded TensorFlow model, including its parameters.   |
| `/model/models`     | GET    | Lists the available TensorFlow models that users can switch to, indicating the default model.      |
| `/model/predict`    | POST   | Classifies an image uploaded by the user as either "damaged" or "not damaged".                     |
| `/model/change`     | POST   | Changes the TensorFlow model used by the server to a specified model.                              |
| `/model/summary`    | GET    | Provides a textual summary of the currently loaded TensorFlow model's architecture.                |
| `/help`             | GET    | Provides an overview and usage examples for the available API endpoints.                           |
