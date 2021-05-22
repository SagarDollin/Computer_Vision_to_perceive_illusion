import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

def pc_model(image_dim):
    
    
    encoder_input0 = keras.Input(shape=image_dim, name='img')
    e0 = keras.layers.Flatten()(encoder_input0)
    e1 = keras.layers.Dense(512,activation="relu")(e0)
    d0 = keras.layers.Dense(1024, activation="sigmoid")(e1)
    decoder0_output = keras.layers.Reshape(image_dim)(d0)
    encoder0 = keras.Model(encoder_input0,e1,name='encoder0')
    autoencoder0 = keras.Model(encoder_input0,decoder0_output, name='autoencoder0')
    
    
    e1_dash = keras.Input(shape=512, name='e1_dash')
    e2 = keras.layers.Dense(256,activation="relu")(e1_dash)
    d1 = keras.layers.Dense(512, activation="relu")(e2)
    encoder1 = keras.Model(e1_dash,e2,name='encoder1')
    autoencoder1 = keras.Model(e1_dash,d1, name='autoencoder1')
    
    e2_dash = keras.Input(shape=256, name='e1_dash')
    e3 = keras.layers.Dense(64,activation="sigmoid")(e2_dash)
    d2 = keras.layers.Dense(256, activation="relu")(e3)
    encoder2 = keras.Model(e2_dash,e3,name='encoder2')
    autoencoder2 = keras.Model(e2_dash,d2, name='autoencoder2')
    
    
    return encoder0,encoder1,encoder2,autoencoder0,autoencoder1,autoencoder2
    

def train_pc_model(encoder0,encoder1,encoder2,autoencoder0,autoencoder1,autoencoder2,X_train,epochs = 2, batch_size = 50):
    broke = 0 
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    autoencoder0.compile(opt, loss='mse')
    autoencoder1.compile(opt, loss='mse')
    autoencoder2.compile(opt, loss='mse')
    
    X_train_batches = []
    n_batches = X_train.shape[0]/batch_size
    for batch in range(int(n_batches)):
        X_train_batches.append(np.array(X_train[batch*batch_size:(batch+1)*batch_size]))
    
    X_train_batches = np.array(X_train_batches)  
    print(X_train_batches.shape)
    
    i = 0 
    alpha1 = 0.1
    beta = 0.2
    gamma = 0.1
    

    for epoch in range(epochs):
        print("**************************Running EPOCH ",epoch," OF ",epochs,"****************")
        if broke == 0:
            for X_train_batch in X_train_batches:
                print("*********Batch numer, ",i,"of ",n_batches,"****************")
                if(i==0):

                    history0 = autoencoder0.fit(
                      X_train_batch,
                      X_train_batch,
                      epochs=1, 
                      batch_size=32, validation_split=0.10
                        )

                    e1_forwardpass = encoder0.predict(X_train_batch)

                    history1 = autoencoder1.fit(
                      e1_forwardpass,
                      e1_forwardpass,
                      epochs=1, 
                      batch_size=32, validation_split=0.10
                        )



                    e2_forwardpass = encoder1.predict(e1_forwardpass)

                    history2 = autoencoder2.fit(
                      e2_forwardpass,
                      e2_forwardpass,
                      epochs=1, 
                      batch_size=32, validation_split=0.10
                        )

                    i+=1

                else:


                    history0 = autoencoder0.fit(
                      X_train_batch,
                      X_train_batch,
                      epochs=1, 
                      batch_size=32, validation_split=0.10
                        )
                    e1_forwardpass = encoder0.predict(X_train_batch)
                    d1_current = autoencoder1.predict(e1_forwardpass)
                    e1_previous = encoder0.predict(X_train_batch_previous)


                    input1 = beta*e1_forwardpass + 0.1*d1_current + (1-0.1-beta)*e1_previous


                    history1 = autoencoder1.fit(
                      input1,
                      input1,
                      epochs=1, 
                      batch_size=32, validation_split=0.10
                        )

                    e2_forwardpass = encoder1.predict(e1_forwardpass)
                    d2_current = autoencoder2.predict(e2_forwardpass)
                    e2_previous = encoder1.predict(e1_forwardpass_previous)

                    input2 = beta*e2_forwardpass + 0.1*d2_current + (1-0.1-beta)*e2_previous



                    history2 = autoencoder2.fit(
                      input2,
                      input2,
                      epochs=1, 
                      batch_size=32, validation_split=0.10
                        )
                    i+=1
                
                X_train_batch_previous = X_train_batch
                e1_forwardpass_previous = e1_forwardpass
        
        
        
        
    return encoder0,encoder1,encoder2,autoencoder0,autoencoder1,autoencoder2

def predict_encoder(encoder0,encoder1,encoder2,X_train):
    train1 = encoder0.predict(X_train)
    print(train1.shape)
    train2 = encoder1.predict(train1)
    print(train2.shape)
    prediction = encoder2.predict(train2)
    print(prediction.shape)
    
    return prediction.reshape(prediction.shape[0],8,8,1)

def classifier_model(shape_dimension):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64,3,padding="same", activation="relu", input_shape=(shape_dimension)))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Dropout(0.2))

    

    model.add(keras.layers.Conv2D(16, 3, padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(2, activation="softmax"))

    return model