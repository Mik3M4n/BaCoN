#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:23:28 2020

@author: Michi
"""

import argparse
import os
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tf.enable_v2_behavior()
tfd = tfp.distributions
from data_generator import create_test_generator, create_generators
from models import *
#from flags import FLAGS
from utils import DummyHist, plot_hist, str2bool, get_flags

from train import ELBO, my_loss



def load_model_for_test(FLAGS, input_shape, n_classes=5, generator=None, FLAGS_ORIGINAL=None):
    
     
    #print('\n -------- Parameters:')
    #for key,value in FLAGS.items():
    #        print (key,value)
     

     
    
    print('------------ BUILDING MODEL ------------\n')
    
    print('Model n_classes : %s ' %n_classes)
    print('Features shape: %s' %str(generator[0][0].shape))
    print('Labels shape: %s' %str(generator[0][1].shape))
    
    
    
    drop=0.
    model=make_model(model_name = FLAGS.model_name, bayesian=FLAGS.bayesian,
                     drop=drop, n_labels=n_classes, 
                     input_shape=input_shape,
                     k_1 = FLAGS.k1,k_2=FLAGS.k2, k_3=FLAGS.k3,
                     n_dense=FLAGS.n_dense, n_conv=FLAGS.n_conv, swap_axes=FLAGS.swap_axes
                     )
    
    model.build(input_shape=input_shape)
    #print(model.summary())
    
    if FLAGS.fine_tune:
        if not FLAGS.swap_axes:
            dense_dim=4*4*FLAGS.k2
        else:
            if FLAGS.n_conv==3:
                dense_dim=FLAGS.k3
            elif FLAGS.n_conv==5:
                dense_dim=FLAGS.k2

            
        model = make_fine_tuning_model(base_model=model, n_out_labels=n_classes,
                                       dense_dim= dense_dim, bayesian=FLAGS.bayesian, trainable=False )
        model.build(input_shape=input_shape)
    print(model.summary())
    
    if generator is not None:
        print('Computing loss for randomly initialized model...')
        loss_0 = compute_loss(generator, model, bayesian=FLAGS.bayesian)
        print('Loss before loading weights/ %s\n' %loss_0.numpy())
            
    
    print('------------ RESTORING CHECKPOINT ------------\n')
    out_path = FLAGS.models_dir+FLAGS.fname
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr)
    ckpts_path = out_path+'/tf_ckpts'
    #ckpt_name = 'ckpt'
    if FLAGS.fine_tune:
        ckpts_path += '_fine_tuning'
    ckpts_path+='/'
    print('Looking for ckpt in ' + ckpts_path)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, net=model)
    
    #manager = tf.train.CheckpointManager(ckpt, ckpts_path, max_to_keep=2, checkpoint_name=ckpt_name)
    
    latest = tf.train.latest_checkpoint(ckpts_path)
    if latest:
          print("Restoring checkpoint from {}".format(latest))
    else:
        print('Checkpoint not found')
    ckpt.restore(latest)
    #ckpt.restore(manager.latest_checkpoint)
    
    if generator is not None:
        loss_1 = compute_loss(generator, model, bayesian=FLAGS.bayesian)
        print('Loss after loading weights/ %s\n' %loss_1.numpy())   
    
    return model


def compute_loss(generator, model, bayesian=False):
    x_batch_train, y_batch_train = generator[0]
    logits = model(x_batch_train, training=False)
    if bayesian:
            #print('Bayesian case')
            kl = sum(model.losses)/generator.batch_size/generator.n_batches
            loss_0 = ELBO(y_batch_train, logits, kl)
    else:
            #print('Frequentist')
            loss_0 = my_loss(y_batch_train, logits)
    #print('Loss ok')
    return loss_0

  


def print_cm(cm, names, out_path, fine_tune=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    matrix_proportions = np.zeros((len(names),len(names)))
    for i in range(0,len(names)):
        matrix_proportions[i,:] = cm[i,:]/float(cm[i,:].sum())
    #print(matrix_proportions)    
    
    confusion_df = pd.DataFrame(matrix_proportions, 
                            index=names,columns=names)
    plt.figure(figsize=(10,10))
    sns.heatmap(confusion_df, annot=True,
            annot_kws={"size": 12}, cmap='gist_gray_r',
            cbar=False, square=True, fmt='.2f');
                
    #accuracy = np.trace(cm) / np.sum(cm).astype('float')
    #misclass = 1 - accuracy
    #score_str='Accuracy: %s, Misclassified: %s'%(accuracy, misclass)
    plt.ylabel(r'True categories',fontsize=14);
    plt.xlabel(r'Predicted categories',fontsize=14);
    plt.tick_params(labelsize=12);
    
    
    
    #plt.show()
    cm_path = out_path+'/cm'
    if fine_tune:
        cm_path+='_FT'
    cm_path+='.png'
    plt.savefig(cm_path)
    print('Saved confusion matrix at %s' %cm_path)
    
    return matrix_proportions    


def evaluate_accuracy(model, test_generator, out_path, names=None, FLAGS=None):
    acc_total=0
    y_true_tot=[]
    y_pred_tot=[]
    for batch_idx, batch in enumerate(test_generator):
        X, y = batch
        pred = model.predict(X, verbose=0)
        y_pred = tf.argmax(tf.nn.softmax(pred, axis=1), axis=1)
        y_true = tf.argmax(y, axis=1)
        equality_batch = tf.equal(y_pred, y_true)
        accuracy = tf.reduce_mean(tf.cast(equality_batch, tf.float32))
        print('Accuracy on %s batch: %s %%' %(batch_idx, accuracy.numpy()))
        acc_total += accuracy
        y_true_tot.append(y_true)
        y_pred_tot.append(y_pred)
    tot_acc = acc_total/test_generator.n_batches
    print('-- Total accuracy: %s %%' %( tot_acc.numpy()))  
    
    #### CONFUSION MATRIX
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(tf.concat(y_true_tot,axis=0),tf.concat(y_pred_tot,axis=0))
    _ = print_cm(cm, names, out_path, fine_tune=FLAGS.fine_tune)
    
    
    return tot_acc


def predict_bayes_label(median, th_prob=0.5):
  if median[median>th_prob].numpy().sum()==0.:
      pred_label = 99.
  else:
      pred_label = tf.argmax(median).numpy()
  return pred_label


def evaluate_accuracy_bayes(model, test_generator, out_path, num_monte_carlo=50, th_prob=0.5, names=None, FLAGS=None):
    acc_total=0
    y_true_tot=[]
    y_pred_tot=[]
    all_sampled_probas = []
    print('Threshold probability for classification: %s ' %th_prob)
    for batch_idx, batch in enumerate(test_generator):
        X, y = batch
        
        y_true = tf.argmax(y, axis=1)
        
        sampled_logits = tf.stack([model.predict(X, verbose=0)
                          for _ in range(num_monte_carlo)], axis=0)
        sampled_probas = tf.nn.softmax(sampled_logits, axis=-1)
        median_proba = tfp.stats.percentile(sampled_probas, 50, axis=0)
        #median_pred = tf.argmax(median_proba, axis=1)
        median_pred = tf.map_fn(fn=lambda x: predict_bayes_label(x, th_prob=th_prob), elems=median_proba)
        
        equality_batch = tf.equal(tf.cast(median_pred, dtype=tf.int64), y_true)
        accuracy = tf.reduce_mean(tf.cast(equality_batch, tf.float32))        
        
        print('Accuracy on %s batch using median of sampled probabilities: %s %%' %(batch_idx, accuracy.numpy()))
        acc_total += accuracy
        y_true_tot.append(y_true)
        y_pred_tot.append(median_pred)
        all_sampled_probas.append(sampled_probas)
    tot_acc = acc_total/test_generator.n_batches
    print('-- Accuracy on test set using median of sampled probabilities: %s %% \n' %( tot_acc.numpy()))  
    
    all_sampled_probas=tf.concat(all_sampled_probas, axis=1)
    all_preds = tf.argmax(all_sampled_probas, axis=-1)
    all_y_true=tf.concat(y_true_tot, axis=0)
    equality_arr = np.array([ tf.reduce_mean(tf.cast(tf.equal(my_pred, all_y_true), tf.float32)) for my_pred in all_preds])
    
    high = np.percentile(equality_arr, 95)
    low= np.percentile(equality_arr, 5)
    median_arr=np.percentile(equality_arr, 50)
    t_string = 'Median: %s + %s -%s (95%% C.I.), %s samples' %(np.round(median_arr,3), np.round(high-median_arr,3) ,np.round(median_arr-low,3), num_monte_carlo)
    import matplotlib.pyplot as plt
    _ = plt.hist(equality_arr)
    plt.xlabel(r'$Test \: Accuracy: \: Binary classification$', fontsize=15)
    plt.ylabel(r'$\mathrm{Counts}$', fontsize=15)
    plt.title(t_string)
    print(t_string)
    acc_path = out_path+'/accuracy_hist'
    if FLAGS.fine_tune:
        acc_path+='_FT'
    acc_path+='.png' 
    plt.savefig(acc_path)
    print('Saved histogram of accuracy at %s' %acc_path)
    
    #### CONFUSION MATRIX
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(tf.concat(y_true_tot,axis=0),tf.concat(y_pred_tot,axis=0))
    
    if tf.unique(tf.concat(y_pred_tot,axis=0)).y.shape[0]>len(names):
        print('Adding Not classified label')
        names = names+['Not classified']
        
    _ = print_cm(cm, names, out_path, fine_tune=FLAGS.fine_tune)
    
    
    return tot_acc





   
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default='', type=str, required=True)
    parser.add_argument("--TEST_DIR", default=None, type=str, required=False)
    
    parser.add_argument("--n_monte_carlo_samples", default=50, type=int, required=False)
    parser.add_argument("--th_prob", default=0.5, type=float, required=False)
    
    parser.add_argument("--batch_size", default=None, type=int, required=False)
    
    
    parser.add_argument("--add_noise", default=None, type=str2bool, required=False)
    parser.add_argument("--n_noisy_samples", default=None, type=int, required=False)
    parser.add_argument("--add_shot", default=None, type=str2bool, required=False)
    parser.add_argument("--add_sys", default=None, type=str2bool, required=False)
    parser.add_argument("--sigma_sys", default=None, type=float, required=False)
    
    
    
    
    args = parser.parse_args()
    
    print('Reading log from %s ' %args.log_path)
    n_flag_ow=False
    FLAGS = get_flags(args.log_path)
    
    
    try:
        sigma_sys=FLAGS.sigma_sys
    except AttributeError:
        print(' ####  FLAGS.sigma_sys not found! #### \n Probably loading an older model. Using sigma_sys=15')
        FLAGS.sigma_sys=15
    
    
    
    if args.TEST_DIR is not None:
        print('Using data in the directory %s' %args.TEST_DIR)
        FLAGS.TEST_DIR = args.TEST_DIR
    if args.batch_size is not None:
        print('Using batch_size %s' %args.batch_size)
        FLAGS.batch_size = args.batch_size
    if args.add_noise is not None:
        n_flag_ow=True
        FLAGS.add_noise = args.add_noise
    if args.n_noisy_samples is not None:
        n_flag_ow=True
        FLAGS.n_noisy_samples=args.n_noisy_samples
    if args.add_shot is not None:
            n_flag_ow=True
            FLAGS.add_shot=args.add_shot
    if args.add_sys is not None:
            n_flag_ow=True
            FLAGS.add_sys=args.add_sys
    if args.sigma_sys is not None:
            n_flag_ow=True
            FLAGS.sigma_sys=args.sigma_sys
    if n_flag_ow:
        print('Overwriting noise flags. Using n_noisy_samples=%s, add_shot=%s, add_sys=%s, sigma_sys=%s' %(FLAGS.n_noisy_samples, str(FLAGS.add_shot),str(FLAGS.add_sys), FLAGS.sigma_sys))
        
    try:
        swap_axes=FLAGS.swap_axes
    except AttributeError:
        if FLAGS.im_channels>1:
            swap_axes=True
        else:
            swap_axes=False
        FLAGS.swap_axes=swap_axes
        #print(FLAGS.swap_axes)
        print(' ####  FLAGS.swap_axes not found! #### \n Probably loading an older model. Set swap_axes=%s' %str(swap_axes))      
        
        
    print('\n -------- Parameters:')
    for key,value in vars(FLAGS).items():
            print (key,value)
    
    
    
    out_path = FLAGS.models_dir+FLAGS.fname
    
    
    print('------------ CREATING DATA GENERATORS ------------\n')
    test_generator = create_test_generator(FLAGS)
        
    print('------------ DONE ------------\n')
        
    
    
    if FLAGS.swap_axes:
        input_shape = ( int(test_generator.dim[0]), 
                   int(test_generator.n_channels))
    else:
        input_shape = ( int(test_generator.dim[0]), 
                   int(test_generator.dim[1]), 
                   int(test_generator.n_channels))
    print('Input shape %s' %str(input_shape))
    
             
    model_loaded =  load_model_for_test(FLAGS, input_shape, n_classes=test_generator.n_classes,
                                        generator=test_generator)
    
   #test_generator[1]
    
    names=[ test_generator.inv_labels_dict[i] for i in range(len(test_generator.inv_labels_dict.keys()))]
    
    if FLAGS.bayesian:
        _ = evaluate_accuracy_bayes(model_loaded, test_generator, out_path,  
                                    num_monte_carlo = args.n_monte_carlo_samples, th_prob=args.th_prob, 
                                    names=names, FLAGS=FLAGS)
    else:
         _ = evaluate_accuracy(model_loaded, test_generator, out_path,  names=names, FLAGS=FLAGS)
    
    
if __name__=='__main__':
    
    main()