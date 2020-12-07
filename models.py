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

def make_custom_model(    drop=0.5, 
                          n_labels=5, 
                          input_shape=( 100, 4), 
                          padding='valid', 
                          filters=(8, 16, 32),
                          kernel_sizes=(10,5,2),
                          strides=(2,2,1),
                          pool_sizes=(2, 2, 0),
                          strides_pooling=(2, 1, 0),
                          activation=tf.nn.leaky_relu,
                          bayesian=True, 
                          n_dense=1, swap_axes=True, BatchNorm=True
                          ):
    
    n_conv=len(kernel_sizes)
    
    
    if swap_axes:
        print('using 1D layers and %s channels' %input_shape[-1])
        is_1D = True
        flayer =  tf.keras.layers.GlobalAveragePooling1D()

        f_dim_0=input_shape[0]
        f_dim_1=1
        
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
        f_dim_0=input_shape[0]
        f_dim_1=input_shape[1]
        flayer = tf.keras.layers.GlobalAveragePooling2D()
        maxpoolL = tf.keras.layers.MaxPooling2D
        
        if not bayesian:
            clayer = tf.keras.layers.Conv2D
            dlayer = tf.keras.layers.Dense
        else:
            clayer = tfp.layers.Convolution2DFlipout
            dlayer = tfp.layers.DenseFlipout
    
    # (3) Create a sequential model
    inputs = tf.keras.Input(shape=input_shape)
    first_layer=True
                                

    
    
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
        print('Expected output dimension of layer %s: %s' %(conv1.name, f_dim_0))

        if first_layer:
            x = conv1(inputs)
            first_layer=False
        else:
            x=conv1(x)
        
        if ps[0]!=0:
            # Pooling 
            maxPool = maxpoolL(pool_size=ps, strides=spool, padding=padding)
            x=maxPool(x)
            #if not is_1D:
            f_dim_0 = (f_dim_0 - ps[0])/spool[0]+1
            print('Expected output dimension of layer %s: %s' %(maxPool.name, f_dim_0))
        # Batch Normalisation
        if i<n_conv and BatchNorm:
                x=tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)
  
    
    
    # Prepare input for Dense layers
    x=flayer(x)
    
    
    # Dense Layers
    for _ in range(n_dense): 
        x = dlayer(filters[-1], activation=activation)(x)
        # Batch Normalisation
        if  BatchNorm:
            x=tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)
        # Add Dropout 
        if not bayesian and drop!=0.:
            x = tf.keras.layers.Dropout(drop)(x)

 
    # Output Layer
    outL = dlayer(n_labels)
    
    outputs = outL(x)
  
    model = tf.keras.Model(inputs, outputs)

    return model



# ------------------------------------------------------



def make_fine_tuning_model(base_model, input_shape, n_out_labels, dense_dim=0, 
                           bayesian=True, trainable=True, drop=0.5, BatchNorm=True, include_last=False):
    
    inputs = tf.keras.Input(shape=input_shape)
    first_layer=True
    
    
    for layer in base_model.layers[:-1]: # go through until ith layer
        layer.trainable = trainable
        if first_layer:
          x=layer(inputs, training=trainable)
          first_layer=False
        else:
          x=layer(x, training=trainable)
        if not layer.trainable:
            print('Layer ' + layer.name + ' frozen.')
        #print(layer.name)
        #print(layer.trainable)
    if include_last:
        last_layer = base_model.layers[-1]
        last_layer.trainable = trainable
        x=last_layer(x, training=trainable)
        if not last_layer.trainable:
            print('Layer ' + last_layer.name + ' frozen.')
        else:
            print('Layer ' + last_layer.name + ' not frozen.')
        x=tf.nn.softmax(x)
    #print('\nBase model done\n')
    if bayesian and dense_dim>0.:
        denseL = tfp.layers.DenseFlipout(dense_dim)
    elif not bayesian:
        denseL = tf.keras.layers.Dense(dense_dim)
          
    if dense_dim>0:
        if bayesian:
          x=tfp.layers.DenseFlipout(dense_dim)(x)
        else:
          x=tf.keras.layers.Dense(dense_dim)(x)
        #print(denseL.name)
        #print(denseL.trainable)
        if  BatchNorm:
            x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)
        if not bayesian and drop!=0.:
            x = tf.keras.layers.Dropout(drop)(x)
    #else:
      #print('No additional dense layer added')
    if bayesian:
        outL = tfp.layers.DenseFlipout(n_out_labels)
    else:
        outL = tf.keras.layers.Dense(n_out_labels)
    outputs = outL(x, training=True)
    #print(outL.name)
    #print(outL.trainable)
    
  
    model = tf.keras.Model(inputs, outputs)

    #print('\nDone.')
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



# ------------------------------------------------------



def make_unfreeze_model(base_model, input_shape, n_out_labels, dense_dim=0, 
                           bayesian=True, drop=0.5, BatchNorm=True,):
    #print('Making unfrozen model')
    inputs = tf.keras.Input(shape=input_shape)
    first_layer=True
    for layer in base_model.layers[:-1]: # go through until last layer
        layer.trainable = True
        #print(layer.name)
        if first_layer:
          x=layer(inputs, training=True)
          first_layer=False
        else:
          x=layer(x, training=True)
        #print(x.shape)
          
    last_layer=base_model.layers[-1]
    last_layer.trainable = True
    outputs=last_layer(x, training=True)
    model = tf.keras.Model(inputs, outputs)

    #print('\nDone.')
    return model



