import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

def convert_rgbtogray(X):
    X_train_gray = []
    for x in X:
        x = tf.image.rgb_to_grayscale(x)
        X_train_gray.append(x)

    return np.array(X_train_gray)
    

def normalize(X):
    return X/255

def cifar_dataset():
    (X_train_cifar,y_train_cifar),(X_test_cifar,y_test_cifar) = keras.datasets.cifar100.load_data()
    #gray
    X_train_cifar = convert_rgbtogray(X_train_cifar)
    X_test_cifar = convert_rgbtogray(X_test_cifar)
    #normalise
    X_train_cifar = normalize(X_train_cifar)
    
    X_test_cifar = normalize(X_test_cifar)
    
    return (X_train_cifar,y_train_cifar),(X_test_cifar,y_test_cifar)

import os



def get_square_pac_data(data_dir,labels):
    data = [] 
    Y = []
    
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        
        for img in os.listdir(path):
            
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                
                data.append(img_arr)
                Y.append(class_num)
                
                
            except Exception as e:
                
                print(e)
    
    data = preprocess_data(np.array(data))
    return data,np.array(Y)

def preprocess_data(data):
    data = convert_rgbtogray(data)
    data = normalize(data)
    return data

def results(epochs_range,history,test):    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy '+test)
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss '+test)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
