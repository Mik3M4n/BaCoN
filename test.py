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
from utils import DummyHist, plot_hist, str2bool, get_flags

from train import ELBO, my_loss


def load_model_for_test(FLAGS, input_shape, n_classes=5,
                        generator=None, FLAGS_ORIGINAL=None, new_fname=None):
         

    
    print('------------ BUILDING MODEL ------------\n')
    
    print('Model n_classes : %s ' %n_classes)
    if generator is not None:
        print('Features shape: %s' %str(generator[0][0].shape))
        print('Labels shape: %s' %str(generator[0][1].shape))
    
    try:
        BatchNorm=FLAGS.BatchNorm
    except AttributeError:
        print(' ####  FLAGS.BatchNorm not found! #### \n Probably loading an older model. Using BatchNorm=True')
        BatchNorm=True
    
    model=make_model(  model_name=FLAGS.model_name,
                         drop=0, 
                          n_labels=n_classes, 
                          input_shape=input_shape, 
                          padding='valid', 
                          filters=FLAGS.filters,
                          kernel_sizes=FLAGS.kernel_sizes,
                          strides=FLAGS.strides,
                          pool_sizes=FLAGS.pool_sizes,
                          strides_pooling=FLAGS.strides_pooling,
                          activation=tf.nn.leaky_relu,
                          bayesian=FLAGS.bayesian, 
                          n_dense=FLAGS.n_dense, swap_axes=FLAGS.swap_axes, BatchNorm=BatchNorm
                          )            
            

    
    model.build(input_shape=input_shape)
    #print(model.summary())
    
    if FLAGS.fine_tune:
        
        if FLAGS.add_FT_dense:
            dense_dim=FLAGS.filters[-1]
        else:
            dense_dim=0

        if len(FLAGS.c_1)==1:
            ft_ckpt_name = '_'+('-').join(FLAGS.c_1)+'vs'+('-').join(FLAGS.c_0)
        else:
            ft_ckpt_name=''
        if not FLAGS.dataset_balanced:
            ft_ckpt_name += '_unbalanced'
        else:
            ft_ckpt_name += '_balanced'
        if not FLAGS.trainable and FLAGS.fine_tune:
            ft_ckpt_name+='_frozen_weights'
        else:
            ft_ckpt_name+='_all_weights'
        if FLAGS.include_last:
            ft_ckpt_name+='_include_last'
        else:
            ft_ckpt_name+='_without_last'
        if FLAGS.unfreeze:
            ft_ckpt_name+='_unfrozen'
        
            
        model = make_fine_tuning_model(base_model=model, input_shape=input_shape, n_out_labels=generator.n_classes_out,
                                       dense_dim= dense_dim, bayesian=FLAGS.bayesian, trainable=False, drop=0, BatchNorm=FLAGS.BatchNorm,
                                       include_last=FLAGS.include_last)
        model.build(input_shape=input_shape)
    print(model.summary())
    
    if generator is not None:
        print('Computing loss for randomly initialized model...')
        loss_0 = compute_loss(generator, model, bayesian=FLAGS.bayesian)
        print('Loss before loading weights/ %s\n' %loss_0.numpy())
            
    
    print('------------ RESTORING CHECKPOINT ------------\n')
    if new_fname is None:
        out_path = FLAGS.models_dir+FLAGS.fname
    else:
        out_path = FLAGS.models_dir+new_fname
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr)
    ckpts_path = out_path
    ckpts_path+='/tf_ckpts'
    if FLAGS.fine_tune:
        ckpts_path += '_fine_tuning'+ft_ckpt_name
    ckpts_path+='/'
    print('Looking for ckpt in ' + ckpts_path)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, net=model)
    
    
    latest = tf.train.latest_checkpoint(ckpts_path)
    if latest:
          print("Restoring checkpoint from {}".format(latest))
    else:
        raise ValueError('Checkpoint not found')
        #print('Checkpoint not found')
    ckpt.restore(latest)
    
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
    return loss_0

  


def print_cm(cm, names, out_path, FLAGS):
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = 'serif'
    plt.rcParams["mathtext.fontset"] = "cm"
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
    plt.tick_params(labelsize=18);
    
    
    #plt.show()
    cm_path = out_path+'/cm'
    if FLAGS.fine_tune:
        cm_path+='_FT'
    if not FLAGS.trainable:
        cm_path+='_frozen_weights'
    
    cm_path_vals = cm_path+'_values.txt'
    cm_path+='.pdf'
    plt.savefig(cm_path)
    np.savetxt(cm_path_vals, cm)
    print('Saved confusion matrix at %s' %cm_path)
    print('Saved confusion matrix values at %s' %cm_path_vals)
    
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
    _ = print_cm(cm, names, out_path, FLAGS)
    
    
    return tot_acc


def predict_bayes_label(mean_prob, th_prob=0.5):
  if type(mean_prob) is not np.ndarray:
      mean_prob=mean_prob.numpy()
  if mean_prob[mean_prob>th_prob].sum()==0.:
      pred_label = 99
  else:
      pred_label = tf.argmax(mean_prob).numpy()
  return pred_label


def predict_mean_proba(X, model, num_monte_carlo=100, softmax=True, verbose=False):
        sampled_logits = tf.stack([model.predict(X, verbose=0)
                          for _ in range(num_monte_carlo)], axis=0)
        if softmax:
            sampled_probas = tf.nn.softmax(sampled_logits, axis=-1)
        else:
            sampled_probas= sampled_logits
        if verbose:
            print("sampled_probas shape: %s" %str(sampled_probas.shape))
        mean_proba=tf.reduce_mean(sampled_probas, axis=0)
        if verbose:
            print("mean_proba  shape: %s" %str(mean_proba.shape))
        return mean_proba, sampled_probas


def my_predict(X, model, num_monte_carlo=100, th_prob=0.5, verbose=False):
  if verbose:
      print('using th_prob=%s'%th_prob)
  mean_proba, sampled_probas = predict_mean_proba(X, model, num_monte_carlo=num_monte_carlo, verbose=verbose)
  mean_pred = tf.map_fn(fn=lambda x: predict_bayes_label(x, th_prob=th_prob), elems=mean_proba)
  if verbose:
      print("mean_pred  shape: %s" %str(mean_pred.shape))
  return sampled_probas, mean_proba, mean_pred 


def evaluate_accuracy_bayes(model, test_generator, out_path, num_monte_carlo=50, th_prob=0.5, names=None, FLAGS=None):
    acc_total=0
    acc_total_no_uncl=0
    y_true_tot=[]
    y_pred_tot=[]
    all_sampled_probas = []
    print('Threshold probability for classification: %s ' %th_prob)
    for batch_idx, batch in enumerate(test_generator):
        X, y = batch
        y_true = tf.argmax(y, axis=1)
        
        # Predict mean probability in each class by averaging on MC samples, then  label with prob threshold
        sampled_probas, mean_proba, mean_pred = my_predict(X, model, num_monte_carlo, th_prob)
        
        # Compute accuracy
        equality_batch = tf.equal(tf.cast(mean_pred, dtype=tf.int64), y_true)
        accuracy = tf.reduce_mean(tf.cast(equality_batch, tf.float32))  
        
        equality_batch_no_uncl = tf.equal(tf.cast(mean_pred[mean_pred!=99], dtype=tf.int64), y_true[mean_pred!=99])
        accuracy_no_uncl = tf.reduce_mean(tf.cast(equality_batch_no_uncl, tf.float32))  
        
        print('Accuracy on %s batch using median of sampled probabilities: %s %%' %(batch_idx, accuracy.numpy()))
        print('Accuracy on %s batch using median of sampled probabilities, not considering unclassified examples: %s %%' %(batch_idx, accuracy_no_uncl.numpy()))
        acc_total += accuracy
        acc_total_no_uncl+=accuracy_no_uncl
        y_true_tot.append(y_true)
        y_pred_tot.append(mean_pred)
        all_sampled_probas.append(sampled_probas)
    tot_acc = acc_total/test_generator.n_batches
    tot_acc_no_uncl = acc_total_no_uncl/test_generator.n_batches
    print('-- Accuracy on test set using median of sampled probabilities: %s %% \n' %( tot_acc.numpy()))
    print('-- Accuracy on test set using median of sampled probabilities, not considering unclassified examples: %s %% \n' %( tot_acc_no_uncl.numpy()))
        
    
    #### CONFUSION MATRIX
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(tf.concat(y_true_tot,axis=0),tf.concat(y_pred_tot,axis=0))
    
    if tf.unique(tf.concat(y_pred_tot,axis=0)).y.shape[0]>len(names):
        print('Adding Not classified label')
        names = names+['Not classified']
        
    _ = print_cm(cm, names, out_path, FLAGS)
    
    
    return tot_acc





   
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default='', type=str, required=True)
    parser.add_argument("--TEST_DIR", default='data/test_data/', type=str, required=False)
    parser.add_argument("--models_dir", default='models/', type=str, required=False)
    
    parser.add_argument("--n_monte_carlo_samples", default=100, type=int, required=False)
    parser.add_argument("--th_prob", default=0.5, type=float, required=False)
    
    parser.add_argument("--batch_size", default=None, type=int, required=False)
    
    parser.add_argument("--add_noise", default=None, type=str2bool, required=False)
    parser.add_argument("--n_noisy_samples", default=None, type=int, required=False)
    parser.add_argument("--add_shot", default=None, type=str2bool, required=False)
    parser.add_argument("--add_sys", default=None, type=str2bool, required=False)
    parser.add_argument("--sigma_sys", default=None, type=float, required=False)
    
    parser.add_argument("--save_indexes", default=False, type=str2bool, required=False)
    
    
    
    
    
    args = parser.parse_args()
    
    print('Reading log from %s ' %args.log_path)
    n_flag_ow=False
    FLAGS = get_flags(args.log_path)
    
     
    if args.save_indexes is not None:
        print('Setting save_indexes to %s' %args.save_indexes)
        FLAGS.save_indexes = args.save_indexes
    if args.TEST_DIR is not None:
        print('Using data in the directory %s' %args.TEST_DIR)
        FLAGS.TEST_DIR = args.TEST_DIR
    if args.models_dir is not None:
        print('Reading model from the directory %s' %args.models_dir)
        FLAGS.models_dir = args.models_dir
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
    
             
    model_loaded =  load_model_for_test(FLAGS, input_shape, n_classes=test_generator.n_classes_out,
                                        generator=test_generator)
    
    
    names=[ test_generator.inv_labels_dict[i] for i in range(len(test_generator.inv_labels_dict.keys()))]
    
    if FLAGS.bayesian:
        _ = evaluate_accuracy_bayes(model_loaded, test_generator, out_path,  
                                    num_monte_carlo = args.n_monte_carlo_samples, th_prob=args.th_prob, 
                                    names=names, FLAGS=FLAGS)
    else:
         _ = evaluate_accuracy(model_loaded, test_generator, out_path,  names=names, FLAGS=FLAGS)
    
    
if __name__=='__main__':
    
    main()