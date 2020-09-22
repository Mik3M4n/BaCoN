#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:17:46 2020

@author: Michi
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse




def get_fname_list(c_0, c_1, list_IDs_temp, data_root):
    
    # np.array([self.data_root + '/'+l+ '/'+ str(ID) + '.txt' for ID in list_IDs_temp for l in self.labels])
    
    fnames_c0 = np.array([ data_root+'/'+c_0[0]+'/'+str(ID)+'.txt' for ID in list_IDs_temp ])
    
    fnames_c1 = []
    n_loops=len(list_IDs_temp)//len(c_1)
    print('get_fname_list n_loops: %s' %n_loops)
    for k in range(n_loops):
        p  = np.random.permutation(list_IDs_temp[k*len(c_1):(k+1)*len(c_1)])
        for i, l in enumerate(c_1):
            fname = data_root+'/'+l+'/'+str(p[i])+'.txt'
            #print(fname)
            fnames_c1.append(fname)
    fnames_c1 = np.array(fnames_c1)
    fname_list = np.concatenate([fnames_c0,fnames_c1])
    
    return fname_list



class DummyHist(object):
  def __init__(self, hist):
    self.history=hist


class dataHolder:
    def __init__(self,**kwargs):
        
        self.__dict__.update(kwargs)
    def print_data(self):
        for i in self.__dict__:
            print(i, ":", self.__dict__[i] )


class DummyFlags:
   def __init__(self, dictionary):
     for k, v in dictionary.items():
        setattr(self, k, v)
        
        

def plot_hist(history, epochs=15, plot_val=True, save=False, path=None, show=False):

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss=history.history['loss']
  val_loss=history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(18, 4))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  if plot_val:
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  if plot_val:
    plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  if save:
    print('Saving at %s' %path)
    plt.savefig(path)
  if show:
   plt.show()
  
  
  
def save_params(model, fname, params):
      fname_par = fname+'_params.json'
      import json
      with open(fname_par, 'w') as fp:
          json.dump(params, fp)

      print('Done.')

import pickle
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def save_model(model, fname, params):

  model.save(fname)
  
  # serialize model to JSON
  model_json = model.to_json()
  with open(fname+".json", "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(fname+".h5")
  print("Saved model to disk")
 
  save_params(model, fname, params)
  
  
  
def cut_sample(indexes, bs, n_labels=2, n_noise=1, Verbose=False):
  if Verbose:
    print('N_labels: %s' %n_labels)
    print('N_noise: %s' %n_noise)
  a = indexes.shape[0]
  if Verbose:
    print('Indexes length: %s' %a)
  n_keep = a - (a % (bs//(n_labels*n_noise)))
  if Verbose:
    print('n_keep: %s' %n_keep)
    
  #print('len(n_keep)/n_labels/n_noise=%s'%(n_keep/n_labels/n_noise))
  if n_keep<a or not((n_keep/n_labels/n_noise).is_integer()):   
    if Verbose:
      print('Sampling')
    idxs_new = np.random.choice(indexes, int(n_keep), replace=False)
  else:   
    if Verbose:
      print('Not sampling')
    idxs_new = indexes
  check_val=int(idxs_new.shape[0]%(bs/(n_labels*n_noise)))
  if Verbose:
    print('New length: %s' %idxs_new.shape[0])
    print('idxs_new.shape0 mod  bs/ n_labels x n_noise : %s' %check_val )
    print('len(idxs_new)/n_noise=%s'%(idxs_new.shape[0]/n_noise))
  if check_val!=0 or not((idxs_new.shape[0]/n_noise).is_integer()) : #(n_labels*n_noise)%idxs_new.shape[0] !=0:
    if Verbose:
      print('Recursive call')
    return(cut_sample(np.random.choice(indexes, int(n_keep-1), replace=False), bs, n_labels=n_labels, n_noise=n_noise, Verbose=Verbose))

  return np.unique(idxs_new)
  


class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        self.write('Prova Logger')
       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__
 
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print('Got v with type:')
        print(type(v))
        print('Value: %s' %v)
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
class Unbuffered(object):

   def __init__(self, stream, te):

       self.stream = stream
       self.te=te

   def write(self, data):

       self.stream.write(data)
       self.stream.flush()
       self.te.write(data)    # Write the data of stdout here to a text file as well
       
   def flush(self):
        self.stream.flush()  
        
        
def parse_flags(FLAGS):
  for key in FLAGS.keys():
    if type(FLAGS[key])!=list:
        
        try:
            FLAGS[key] = int( FLAGS[key])
        except ValueError:
            try: 
                FLAGS[key] = float( FLAGS[key])
            except ValueError:
                # can be str or bool 
                if FLAGS[key].lower() in ('yes', 'true', 't', 'y', '1'):
                    FLAGS[key]= True
                elif FLAGS[key].lower() in ('no', 'false', 'f', 'n', '0'):
                    FLAGS[key]= False
                else:
                    # it is a string
                    if FLAGS[key]=='None':
                        FLAGS[key]=None
                    
                    #or_string=FLAGS[key]
                    #print(or_string)
                    #parsed_str = or_string.replace(' ', '\ ' )
                    #print(parsed_str)
                    FLAGS[key]= FLAGS[key]
    else:
        FLAGS[key]= FLAGS[key]
  return FLAGS          
 
def not_start(line): 
  if ' -------- Parameters:' in line and not 'Skipping' in line:
    return False
  else:
    return True
         
def get_flags(log_path):
    from itertools import dropwhile
    FLAGS={}
    with open(log_path) as f: 
        for line in dropwhile(not_start, f):
            #first_line = next(f)
            #print('Skipping %s' %first_line)
        #header_line = next(f)
        #print('Skipping %s' %header_line)
        #for line in f:
            if '------------ CREATING DATA GENERATORS ------------' in line or not line.strip():
                break
            else:
                #print(line)
                if line.split()[0]=='models_dir':
                    key, value = line.split()[0], line.split()[1]+' '+line.split()[2]
                            
                elif line.split()[0]=='log_path':
                    # This case corresponds to log_path
                    key, value = line.split()[0], ''
                elif line.split()[0] in ('z_bins', 'filters', 'kernel_sizes', 'strides', 'pool_sizes', 'strides_pooling') :
                    key, value = line.split()[0], [int(line.split()[1:][i].strip('\[,\' \]')) for i in range(len(line.split()[1:])) ]
                
                elif line.split()[0] in ('c_0', 'c_1'):
                    key, value = line.split()[0], [line.split()[1:][i].strip('\[,\' \]') for i in range(len(line.split()[1:]))]
                elif len(line.split())==2:
                    key, value = line.split()[0], line.split()[1] 
                else:
                    print('The following line has not been formatted: \n %s' %line)
                    key, value = line.split()[0], line.split()[1] 
                FLAGS[key]=value
    
    FLAGS.pop('--------')
    FLAGS = parse_flags(FLAGS)
    if 'c_1' in FLAGS.keys():
        if len(FLAGS['c_1'])>1:
            fine_tune_dict={ label:'non_lcdm' for label in FLAGS['c_1']}
        else:
            fine_tune_dict={ label:label for label in FLAGS['c_1']}
        
        #fine_tune_dict={ label:'non_lcdm' for label in FLAGS['c_1']}
        FLAGS['fine_tune_dict'] = fine_tune_dict
        for i in range(len(FLAGS['c_0']) ):
          FLAGS['fine_tune_dict'][FLAGS['c_0'][i]]=FLAGS['c_0'][i]
        
    print('\n -------- Loaded parameters:')
    for key,value in FLAGS.items():
        print (key,value)
  
    FLAGS_DH = DummyFlags(FLAGS)
  
    return FLAGS_DH


def get_all_indexes(FLAGS, Test=False):
    
    if not Test:
        data_dir = FLAGS.DIR
    else:
        data_dir = FLAGS.TEST_DIR
    all_labels =  ([name for name in os.listdir(data_dir) if not os.path.isfile(os.path.join(data_dir, name))])
    

    
    if not FLAGS.fine_tune:
        labels=all_labels
        if FLAGS.sort_labels:
            labels.sort()
        labels_dict = {labels[i]:int(i) for i in range(len(labels))}
    else:
        if len(FLAGS.c_1)>1:
            c_1_class_name='non_lcdm'
            labels=['lcdm', c_1_class_name]
        else:
            c_1_class_name=FLAGS.c_1[0]
            labels = [l for l in FLAGS.c_1]+[l for l in FLAGS.c_0]
            if not all(elem in all_labels  for elem in labels):
                raise ValueError('Specified labels for fine-tuning are not in the dataset!')
        #labels_dict={ label:1 for label in FLAGS.c_1}
        #for i in range(len(FLAGS.c_0) ):
        #  labels_dict[FLAGS.c_0[i]]=0 #FLAGS.c_0[i]
        labels_dict = {'lcdm':0, c_1_class_name:1}
    
    print('labels : %s' %labels)
    print('Labels encoding: ')
    print(labels_dict)
    n_labels=len(np.unique([val for val in labels_dict.values()]))
    print('n_labels : %s' %n_labels)
    n_s=[]
    for l in all_labels:
        if not Test:
            dir_name=data_dir+'/'+l
        else:
            dir_name=data_dir+'/'+l #+'_test'
        n_samples = len([name for name in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, name))]) 
        print('%s - %s training examples' %(l,n_samples))
        n_s.append(n_samples)

    for i in range(len(all_labels)-1):
        assert n_s[i] == n_s[i+1]
    
    n_samples = n_s[0]
    
    l = 'lcdm'
    if not Test:
        dir_name=data_dir+'/'+l
    else:
        dir_name=data_dir+'/'+l #+'_test'
    all_index = np.array([int(str.split(name, sep='.')[0]) for name in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, name))])
    assert all_index.shape[0]==n_samples
    print('\nN. of data files: %s' %all_index.shape)
    if FLAGS.test_mode and not Test:
        print('Choice with seed %s ' %FLAGS.seed)
        np.random.seed(FLAGS.seed)
        if FLAGS.fine_tune:
            my_size = 2*len(FLAGS.c_1)
        else:
            my_size = FLAGS.n_test_idx
        all_index = np.random.choice(all_index, size=my_size, replace=False)
        val_size = 0.5
    elif not Test:
        val_size = FLAGS.val_size
    else:
        val_size=None
    n_samples=all_index.shape[0]
    

        
    print('get_all_indexes labels dict: %s' %str(labels_dict)) 
    return all_index, n_samples, val_size, n_labels, labels, labels_dict



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]