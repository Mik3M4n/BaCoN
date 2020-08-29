#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:20:20 2020

@author: Michi
"""
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tf.enable_v2_behavior()
tfd = tfp.distributions


def make_model(model_name, **params):
    
    if model_name=='AlexNet': 
        #if params['bayesian']:
            #print('Using make_model_AlexNet_bayes...')
            #return make_model_AlexNet_bayes(**params)
        return make_model_AlexNet(**params)
    elif model_name=='dummy':
        return make_model_dummy(**params)
    else:
        raise ValueError('Enter a valid model name.')



def make_model_AlexNet(   drop=0.5, 
                          n_labels=5, 
                          input_shape=( 125, 1, 4), 
                          padding='valid', 
                          k_1=8, k_2 = 16, k_3  =32,
                          activation=tf.nn.leaky_relu,
                          bayesian=False, 
                          n_dense=0, n_conv=3, swap_axes=True
                          ):
    
    if swap_axes:#input_shape[-1]>1 :
        print('using 1D layers and %s channels' %input_shape[-1])
        is_1D = True
        flayer =  tf.keras.layers.GlobalAveragePooling1D()
        f_dim=1
        
        maxpoolL = tf.keras.layers.MaxPooling1D
        
        if not bayesian:
            clayer = tf.keras.layers.Conv1D
            dlayer = tf.keras.layers.Dense
        else:
            clayer = tfp.layers.Convolution1DFlipout
            dlayer = tfp.layers.DenseFlipout
    else:
        is_1D=False
        f_dim=4*4
        flayer = tf.keras.layers.Flatten()
        maxpoolL = tf.keras.layers.MaxPooling2D
        
        if not bayesian:
            clayer = tf.keras.layers.Conv2D
            dlayer = tf.keras.layers.Dense
        else:
            clayer = tfp.layers.Convolution2DFlipout
            dlayer = tfp.layers.DenseFlipout
    
  # (3) Create a sequential model
    model = tf.keras.models.Sequential() 
                                
    model.add(tf.keras.Input(shape=input_shape))

    # 1st Convolutional Layer
    
    ks1=(11,1)
    st1=(2,1)
    ps1=(2,1)
    spool1=(2,1)
    if is_1D:
        ks1, st1, ps1, spool1= (ks1[0],), (st1[0],), (ps1[0],), (spool1[0],)

    conv1 = clayer(filters=k_1, input_shape=input_shape, 
                                  kernel_size=ks1, strides=st1, 
                                  padding=padding, activation=activation)


    model.add(conv1)
    # Pooling 
    model.add(maxpoolL(pool_size=ps1, strides=spool1, padding=padding))
    # Batch Normalisation before passing it to the next layer
    model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99))
  
  
    # 2nd Convolutional Layer
    ks2=(5,1)
    st2=(2,1)
    ps2=(2,1)
    spool2=(1,1)
    if is_1D:
        ks2, st2, ps2, spool2= (ks2[0],), (st2[0],), (ps2[0],), (spool2[0],)

    conv2 = clayer(filters=k_2, 
                                  kernel_size=ks2, strides=st2, 
                                  padding=padding, activation=activation)
                                  

    model.add(conv2)
    # Pooling
    model.add(maxpoolL(pool_size=ps2, strides=spool2, padding=padding))
    # Batch Normalisation
    model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99))

  
    # 3rd Convolutional Layer
    ks3=(2,1)
    st3=(1,1)
    if is_1D:
        ks3, st3=(ks3[0],), (st3[0],)
    conv3 = clayer(filters=k_3, 
                                  kernel_size=ks2, strides=st3, 
                                  padding=padding, activation=activation)
    model.add(conv3)
#    if n_conv==3:
#        ps3=(2,1)
#        spool3=(2,1)
#        if is_1D:
#            ps3, spool3= (ps3[0],), (spool3[0],)
#        model.add(maxpoolL(pool_size=ps3, strides=spool3, padding=padding))
    # Batch Normalisation
#    model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99))

    if n_conv==5:
        
        ps3=(2,1)
        spool3=(2,1)
        if is_1D:
            ps3, spool3= (ps3[0],), (spool3[0],)
        model.add(maxpoolL(pool_size=ps3, strides=spool3, padding=padding))
        # Batch Normalisation
        model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99))
        
        
        # 4th Convolutional Layer
        ks4=(2,1)
        st4=(1,1)
        if is_1D:
            ks4, st4=(ks4[0],), (st4[0],)
        conv4 = clayer(filters=k_3, 
                                  kernel_size=ks4, strides=st4, 
                                  padding=padding, activation=activation)

        model.add(conv4)
        # Batch Normalisation
        model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99))

        # 5th Convolutional Layer
        ks5=(2,1)
        st5=(1,1)
        ps5=(2,1)
        spool5=(2,1)
        if is_1D:
            ks5, st5, ps5, spool5=(ks5[0],), (st5[0],), (ps5[0],), (spool5[0],)
        conv5 = clayer(filters=k_2, 
                                  kernel_size=ks5, strides=st5, 
                                  padding=padding, activation=activation)
  
        model.add(conv5)
        # Pooling
        #model.add(maxpoolL(pool_size=ps5, strides=spool5, padding=padding))
        
        
    
    elif n_conv!=3:
        raise ValueError('This architecture for the moment only supports n_conv=3 or n_conv=5')
    # Batch Normalisation
    model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99))
    
    # Passing it to a dense layer
    model.add(flayer)
    
    
    #if n_conv==3:
    #        dense_dim=k_3
    #elif n_conv==5:
    #        dense_dim=k_2
    dense_dim=k_2
    # Dense Layers
    for _ in range(n_dense): 
        
        model.add(dlayer(f_dim* dense_dim, activation=activation))
        # Batch Normalisation
        model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99))
        # Add Dropout 
        if not bayesian and drop!=0.:
            model.add(tf.keras.layers.Dropout(drop))

 
    # Output Layer
    outL = dlayer(n_labels)
    
    model.add(outL)

    return model







# ------------------------------------------------------


def make_fine_tuning_model(base_model, n_out_labels, dense_dim=32, bayesian=True, trainable=True):

    model = tf.keras.models.Sequential()

    for layer in base_model.layers[:-1]: # go through until ith layer
        model.add(layer)

    if bayesian and dense_dim>0.:
        denseL = tfp.layers.DenseFlipout(dense_dim)
    elif not bayesian:
        denseL = tf.keras.layers.Dense(dense_dim)
    if dense_dim>0:
        model.add(denseL)
    
    if bayesian:
        outL = tfp.layers.DenseFlipout(n_out_labels)
    else:
        outL = tf.keras.layers.Dense(n_out_labels)
    model.add(outL)

    return model



# ------------------------------------------------------


def make_model_dummy(drop=0., 
                          n_labels=5, 
                          input_shape=( 125, 4, 1), 
                          padding='valid', 
                          k_1=96, k_2 = 256, k_3  =384,
                          activation=tf.nn.leaky_relu,
                          bayesian=False
                          ):
  
  # (3) Create a sequential model
  model = tf.keras.models.Sequential() 
                                      

  model.add(tf.keras.Input(shape=input_shape))

  # 1st Convolutional Layer
  if bayesian:
      
      c1 = tfp.layers.Convolution2DFlipout(filters=15, input_shape=input_shape, kernel_size=(11,1), 
                         strides=(2,1), padding=padding, activation=activation
                         )
  else:
      c1 =  tf.keras.layers.Conv2D(filters=15, input_shape=input_shape, kernel_size=(11,1), 
                         strides=(2,1), padding=padding, activation=activation),
  model.add(c1)
   #Pooling 
  model.add(tf.keras.layers.GlobalAveragePooling2D())
  # Batch Normalisation before passing it to the next layer
  #tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99),
  


  # Output Layer
  if bayesian:
      outL=tfp.layers.DenseFlipout(n_labels)
  else:
      outL=tf.keras.layers.Dense(n_labels) 
  
  model.add(outL)

  return model

