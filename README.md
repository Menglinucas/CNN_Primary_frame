Primary_frame_CNN（dog_cat.py, dataset.py）  
============
A primary frame of Convolutional Neural Network with a clear designing process, mainly used for object classification.  
------------
![image](https://github.com/Menglinucas/Primary_frame_CNN/blob/master/CNN.PNG)  

## A: dog and cat --- dog_cat.py
### 1. Load the training data  
### 2. Define basic parameters  
### 3. Construct networks  
> #### 3.1 Construct layer: features, variables  
> #### 3.2 Construct linkers: filters, weight and bias  
> #### 3.3 Construct operation  
> #### 3.4 Construct loss function  
> #### 3.5 Construct optimizer function used for updating weight and bias  
> #### 3.6 Construct evaluation function used for estimating the accuracy  
### 4. Training  

## B: MNIST --- simple minist.py  
************************************************************************************  
************************************************************************************  
Further reading  
===============  
## A: [About CNN variants](https://www.cnblogs.com/skyfsm/p/6806246.html)  
> ### R-CNN: Region based CNN  
   * box-selecting method: EdgeBoxes, Selective Search, ect.  
> ### SPP: Spatial Pyramid Pooling  
   * used for multi-scale input for CNN  
> ### Fast R-CNN  
   * RCNN + SPP  
> ### Faster R-CNN  
   * add **RPN (Region Proposal Network)** for extracting candidate boxes  
  
  
## B: [25 Neural Network Model](http://blog.csdn.net/qq_35082030/article/details/73368962)  
![image](https://github.com/Menglinucas/Primary_frame_CNN/blob/master/NN.jpg)

400 rows of classic CNN code（classic CNN 400 rows.py）  
============
### function for constructing Graph, training, testing, loading model and classifying
### continue training
### tf.name_scope, get_tensor_by_name, get_operation_by_name
### graph and session, interactiveSession
### tensorboard
### model file formats: ckp, pb
