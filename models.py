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
    
    if model_name=='dummy':
        return make_model_dummy(**params)
    elif model_name=='custom':
        return make_custom_model(**params)
    else:
        raise ValueError('Enter a valid model name.')


# ------------------------------------------------------

def make_custom_model(   drop=0.5, 
                          n_labels=2, 
                          input_shape=( 125, 4, 1), 
                          padding='valid', 
                          filters=(8, 16, 32),
                          kernel_sizes=(10,5,2),
                          strides=(2,2,1),
                          pool_sizes=(2, 2, 1),
                          strides_pooling=(2, 1, 1),
                          activation=tf.nn.leaky_relu,
                          bayesian=False, 
                          n_dense=1, swap_axes=True, BatchNorm=True
                          ):
    
    n_conv=len(kernel_sizes)
    
    if swap_axes:#input_shape[-1]>1 :
        print('using 1D layers and %s channels' %input_shape[-1])
        is_1D = True
        flayer =  tf.keras.layers.GlobalAveragePooling1D()
        #flayer = tf.keras.layers.Flatten()
        #f_dim=1
        f_dim_0=input_shape[0]
        f_dim_1=1#input_shape[1]
        
        maxpoolL = tf.keras.layers.MaxPooling1D
        
        if not bayesian:
            clayer = tf.keras.layers.Conv1D
            dlayer = tf.keras.layers.Dense
        else:
            clayer = tfp.layers.Convolution1DFlipout
            dlayer = tfp.layers.DenseFlipout
    else:
        print('using 2D layers and %s channels' %input_shape[-1])
        is_1D=False
        #f_dim=1
        f_dim_0=input_shape[0]
        f_dim_1=input_shape[1]
        #flayer = tf.keras.layers.Flatten()
        flayer = tf.keras.layers.GlobalAveragePooling2D()
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

    
    
    for i in range(n_conv):
    
    #  Convolutional Layers
    
        ks=(kernel_sizes[i],1)
        st=(strides[i],1)
        ps=(pool_sizes[i],1)
        spool=(strides_pooling[i],1)
        if is_1D:
            ks, st, ps, spool= (ks[0],), (st[0],), (ps[0],), (spool[0],)

        conv1 = clayer(filters=filters[i], input_shape=input_shape, 
                                  kernel_size=ks, strides=st, 
                                  padding=padding, activation=activation)
        #if not is_1D:
        f_dim_0 = (f_dim_0 - ks[0])/st[0]+1
        print(f_dim_0)

        model.add(conv1)
        if ps[0]!=0:
            # Pooling 
            model.add(maxpoolL(pool_size=ps, strides=spool, padding=padding))
            #if not is_1D:
            f_dim_0 = (f_dim_0 - ps[0])/spool[0]+1
            print(f_dim_0)
        # Batch Normalisation
        if i<n_conv and BatchNorm:
                model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99))
  
    
    
    # Prepare input for Dense layers
    model.add(flayer)
    
    
    # Dense Layers
    #if not is_1D:
    f_dim = f_dim_0*f_dim_1
    for _ in range(n_dense): 
        model.add(dlayer(filters[-1], activation=activation))
        # Batch Normalisation
        if  BatchNorm:
            model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99))
        # Add Dropout 
        if not bayesian and drop!=0.:
            model.add(tf.keras.layers.Dropout(drop))

 
    # Output Layer
    outL = dlayer(n_labels)
    
    model.add(outL)

    return model





# ------------------------------------------------------


def make_fine_tuning_model(base_model, n_out_labels, dense_dim=32, 
                           bayesian=True, trainable=True, drop=0.5, BatchNorm=True):

    model = tf.keras.models.Sequential()
    for layer in base_model.layers[:-1]: # go through until ith layer
        model.add(layer)

    if bayesian and dense_dim>0.:
        denseL = tfp.layers.DenseFlipout(dense_dim)
    elif not bayesian:
        denseL = tf.keras.layers.Dense(dense_dim)
    
    if dense_dim>0:
        model.add(denseL)
        if  BatchNorm:
            model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99))
        if not bayesian and drop!=0.:
            model.add(tf.keras.layers.Dropout(drop))
    
    if bayesian:
        outL = tfp.layers.DenseFlipout(n_out_labels)
    else:
        outL = tf.keras.layers.Dense(n_out_labels)
    model.add(outL)
    
    if dense_dim>0:
        fine_tune_at=-2
    else:
         fine_tune_at=-1
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = trainable
    
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

