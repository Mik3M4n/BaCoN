#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:07:43 2020

@author: Michi
"""
import os
import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

from utils import cut_sample, get_all_indexes, get_fname_list, find_nearest


def generate_noise_old(k, P, V=4, delta_k = 0.055):
  sigma_hat_noise = (2*np.pi/((k)**(3/2)*np.sqrt(V*(1e3)**3*delta_k))).T
  sigma_noise = np.abs(P*sigma_hat_noise[:,None])
  return sigma_noise


def generate_noise(k, P, 
                   add_shot=True,
                   add_sys=True,
                   V=np.array([10.43, 6.27, 3.34, 0.283]), 
                   nbar=np.array([0.000358, 0.000828, 0.00103, 0.00128]),
                   delta_k = 0.055,  sigma_sys=15, quadrature=True):
  
  sigma_hat_noise = (2*np.pi/((k[:, None])*np.sqrt(V*(1e3)**3*delta_k)))
  sigma_noise = np.abs(P*sigma_hat_noise)
  
  if add_shot:
      sigma_noise_shot=(sigma_hat_noise/np.array(nbar))
      sigma_noise = sigma_noise+sigma_noise_shot
  if add_sys:
      if quadrature:
        sigma_noise =np.sqrt(sigma_noise**2+sigma_sys**2 )
      else:
        sigma_noise =sigma_noise+sigma_sys
     
  return sigma_noise


class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, labels_dict, 
                 data_root = 'data/', batch_size=32, dim=(500, 4), n_channels=1,
                shuffle=True, normalization='stdcosmo',
                save_indexes=False, models_dir = 'models/MM/', idx_file_name = '_',
                 norm_data_name='/planck.txt',
                 sample_pace = 4, pad=False, 
                 Verbose=False, 
                 k_max=None, i_max = None,
                  add_noise=True, n_noisy_samples = 1, 
                  add_shot=True, add_sys=True,sigma_sys=15,
                  fine_tune = False, 
                  c_0=None, c_1=None, fine_tune_dict=None, 
                  z_bins=[0, 1, 2, 3], swap_axes=False,
                 ):
      
        'Initialization'

        
        self.sigma_sys=sigma_sys
        self.add_shot=add_shot
        self.add_sys=add_sys
        self.fine_tune_dict=fine_tune_dict
        self.fine_tune=fine_tune
        self.c_0=c_0
        self.c_1=c_1
        
                
        self.swap_axes = swap_axes
         
        
        self.k_max=k_max
        self.i_max=i_max
        self.sample_pace=sample_pace
        if sample_pace ==1:
          self.dim = dim
        else:
          self.dim = (int(dim[0]/sample_pace), dim[1]) 
        self.n_channels = n_channels
        self.z_bins=np.asarray(z_bins, dtype=int)
        
        print('Using z bins %s' %z_bins)
        #print(self.z_bins.shape[0])
        if not self.swap_axes:
            if self.z_bins.shape[0]!=self.dim[1]:
                raise ValueError('Number of z bins does not match dimension 1 of the data.')
        else:
            if self.z_bins.shape[0]!=self.n_channels:
                raise ValueError('Number of z bins does not match n_channels.')
        
        self.data_root=data_root
        self.norm_data_path = self.data_root+norm_data_name
        # Select points up to k_max or i_max
        self.all_ks = np.loadtxt(self.norm_data_path)[:, 0]
        if self.sample_pace !=1:
                self.all_ks = np.loadtxt(self.norm_data_path)[0::sample_pace, 0]
        
        #print('Data Gen using k max %s' %str(self.k_max))
        if self.k_max is not None:
            print('Specified k_max is %s' %self.k_max)
            self.i_max, k_max_res = find_nearest(self.all_ks, self.k_max) # self.all_ks[self.all_ks==self.k_max]
            print('Corresponding i_max is %s' %self.i_max)
            print('Closest k to k_max is %s' %k_max_res)
            #print('Selecting ks up to k_max=%s, or index %s for input k_max=%s' %(self.ind_max, k_max_res, self.k_max))
            
        elif self.i_max is not None:
            self.k_max = self.all_ks[self.i_max]
            print('Specified i_max is %s' %self.i_max)
            print('Corresponding k_max is %s' %self.k_max)
            
        elif self.i_max is not None and self.k_max is not None:
            print('Specified i_max is %s' %self.i_max)
            print('Specified k_max is %s' %self.k_max)
            
            i_m, k_m = find_nearest(self.all_ks, self.k_max)
            assert(i_m==self.i_max)
        
        else:
            self.i_max = -1
            print('No cut in k. Using all ks . k_max=%s' %self.all_ks[self.i_max])
            
        
        self.all_ks = self.all_ks[:self.i_max]
        self.dim = (self.all_ks.shape[0], self.dim[1])
        print('New data dim: %s' %str(self.dim) )
        print('Final i_max used is %s' %self.i_max)
            
        
            
        self.batch_size = batch_size
        
        self.labels = labels
        #print(self.labels)
        self.labels_dict = labels_dict
        self.inv_labels_dict={value:key for key,value in zip(self.labels_dict.keys(), self.labels_dict.values())}
        #print(self.inv_labels_dict)
        self.list_IDs = list_IDs
        if len(self.list_IDs)==1:
            self.list_IDs_dict = {label:list_IDs+i for i,label in enumerate(labels)}
            print('Ids dict to use in data gen: %s' %str(self.list_IDs_dict))
        else:
            self.list_IDs_dict = {label:list_IDs for label in labels}
        
        
        self.n_classes = len(self.labels)
        
        self.shuffle = shuffle
        self.on_epoch_end()
        #print('Batch size: %s' %self.batch_size)
        #print('N. samples used for each different label: %s' %self.n_indexes)
        self.save_indexes = save_indexes
        if self.save_indexes:
            self.save_indexes_dict={}
        self.normalization=normalization
        
        if self.normalization=='stdcosmo':
          self.norm_data = np.loadtxt(self.norm_data_path)[:, 1:]
          if self.sample_pace !=1:
            self.norm_data = self.norm_data[0::self.sample_pace, :]
          self.norm_data = self.norm_data[:self.i_max]
              
        
        self.idx_file_name = idx_file_name
        self.models_dir = models_dir
        self.pad=pad
        self.add_noise=add_noise
        if not self.add_noise:
          self.n_noisy_samples = 1
        else:
          self.n_noisy_samples = n_noisy_samples
        
        if self.fine_tune:
            if self.batch_size%(self.n_classes*len(self.c_1)*self.n_noisy_samples):
                raise ValueError('batch size must be multiple of n_classes x len(c_1) x n_noisy_samples')
        else:
            if self.batch_size%(self.n_classes*self.n_noisy_samples):
                raise ValueError('batch size must be multiple of n_classes x n_noisy_samples')
          
        if not self.fine_tune:
            if self.batch_size%(self.n_classes*self.n_noisy_samples)!=0:
                print('Batch size = %s' %self.batch_size)
                print('( n_labels x n_noisy_samples) = %s' %(self.n_classes*self.n_noisy_samples))
                raise ValueError('Batch size must be multiple of (number of classes) x (n_noisy_samples) ')
            self.n_indexes = self.batch_size//(self.n_classes*self.n_noisy_samples) # now many index files to read per each batch
        else:
            if self.batch_size%(self.n_classes*self.n_noisy_samples)!=0:
                print('Batch size = %s' %self.batch_size)
                #print('( n_labels x n_noisy_samples) = %s' %(self.n_classes*self.n_noisy_samples))
                raise ValueError('Batch size must be multiple of 2*(number of classes in non-lcdm) x (n_noisy_samples) ')
            self.n_indexes = self.batch_size//(self.n_classes*self.n_noisy_samples)
            print('batch_size, c_1, n_noisy_samples= %s,%s,%s' %(self.batch_size, len(self.c_1), self.n_noisy_samples))
            print('n_indexes=%s' %self.n_indexes)
            #if self.n_indexes<=len(self.c_0)+len(self.c_1):
            #    print('n_indexes=%s' %self.n_indexes)
            #    print('class 0 labels=%s' %len(self.c_0))
            #    print('class 1 labels=%s' %len(self.c_1))
            #    raise ValueError('Should have an index for each label in non_lcdm ') 
        
        self.n_batches = len(list_IDs)//(self.n_indexes)
        if self.n_batches==0:
          print('list_IDs length: %s' %len(list_IDs))
          print('n_indexes: %s' %self.n_indexes)
          print('batch size: %s' %self.batch_size)
          print('n_batches : %s' %self.n_batches)
          raise ValueError('Not enough examples to support this batch size ')
        print('For each batch we read %s file IDs' %self.n_indexes)
        if not self.fine_tune:
            print('For each file ID we have %s labels' %self.n_classes)
        else:
            print('We read %s IDs for label %s and 1 ID for each of the labels %s' %(str(len(self.c_1)), c_0[0],str( c_1)) )
        if self.add_noise:
          print('For each ID, label we have %s realizations of noise' %self.n_noisy_samples)
         
        if not self.fine_tune:
            n_ex = self.n_indexes*self.n_classes*self.n_noisy_samples
            n_check = self.n_classes*self.n_noisy_samples
        else:
            n_ex = self.n_indexes*self.n_classes*self.n_noisy_samples
            n_check = self.n_classes*self.n_noisy_samples
        print('In total, for each batch we have %s training examples' %(n_ex))
        print('Input batch size: %s' %self.batch_size)
        print('N of batches to cover all file IDs: %s' %self.n_batches)
        
        
        if self.n_indexes%(self.batch_size/(n_check))!=0:
          print('Batch size = %s' %self.batch_size)
          print('( n_labels x n_noisy_samples) = %s' %(n_check*self.n_noisy_samples))
          print('n_indexes = %s' %self.n_indexes)
          raise ValueError('Batch size should satisfy  m x Batch size /( n_labels x n_noisy_samples) =  n_indexes  with m a positive integer ')
        
        if self.n_indexes!=len(list_IDs)/self.n_batches: 
          #print('Batch size = %s' %self.batch_size)
          #print('( n_labels x n_noisy_samples) = %s' %(self.n_classes*self.n_noisy_samples))
          print('length of IDs = %s' %str(len(list_IDs)))
          print('n_batches = %s' %self.n_batches)
          print('n°indexes = %s' %self.n_indexes)
          print('len(list_IDs)/self.n_batches = %s' %(len(list_IDs)/self.n_batches))
          raise ValueError('n_batches does not match length of IDs')
        self.Verbose=Verbose
        #if len(self.list_IDs)==1:
        #    self.Verbose=True
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs)*self.n_classes*self.n_noisy_samples / self.batch_size))
    
    def __shape__(self):
        'I dont know what exactly I should put here - where is n_channels ??? '
        return((len(self.list_IDs), self.dim[0]/self.sample_pace, self.dim[1] ))

    #def shape(self):
    #    return((len(self.list_IDs), self.dim[0]/self.sample_pace, self.dim[1]))    

    def __getitem__(self, index):
        'Generate one batch of data'
        self.batch_idx = index
        
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.n_indexes:(index+1)*self.n_indexes]
        indexes_dict = {self.labels[i]:self.indexes_dict[self.labels[i]][index*self.n_indexes:(index+1)*self.n_indexes] for i, label in enumerate(self.labels)}
        #print('Indexes: %s' %indexes)

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_IDs_temp_dict = { self.labels[i]:[self.list_IDs_dict[self.labels[i]][k] for k in indexes_dict[self.labels[i]]] for i,label in enumerate(self.labels)}

        #print('List_IDs: %s' %list_IDs_temp)

        # Generate data
        #X, y = self.__data_generation(list_IDs_temp)
        X, y = self.__data_generation(list_IDs_temp, list_IDs_temp_dict)
        return  X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        self.indexes_dict = {self.labels[i]:np.arange(len(self.list_IDs_dict[self.labels[i]] )) for i,label in enumerate(self.labels)}
        #print('--- Epoch ended')
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            for label in self.labels:
                np.random.shuffle(self.indexes_dict[label])
            #print('All indexes, permutation: %s' %self.indexes)

 

    def __data_generation(self, list_IDs_temp, list_IDs_temp_dict):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        i_ind = 0
        if self.Verbose:
            print('Dim of X: %s' %str(X.shape))
        # Generate data
        if self.Verbose:
            print('len of list IDs: %s ' %(len(list_IDs_temp)))
        if not self.fine_tune:
            fname_list=[]
            for l in self.labels:
                for ID in list_IDs_temp_dict[l]:
                    t_st =  self.data_root + '/'+l+ '/'+ str(ID) + '.txt' 
                    fname_list.append(t_st)
            fname_list = np.array(fname_list)
            #fname_list = np.array([self.data_root + '/'+l+ '/'+ str(ID) + '.txt' for ID in list_IDs_temp for l in self.labels])
        else:
            #print(list_IDs_temp)
            fname_list = get_fname_list(self.c_0, self.c_1, list_IDs_temp, self.data_root)
        if self.fine_tune and self.Verbose:
            print(fname_list)
        #print('len(fname_list), batch_size, n_noisy_samples: %s, %s, %s' %(len(fname_list), self.batch_size, self.n_noisy_samples))
        assert len(fname_list)==self.batch_size//(self.n_noisy_samples)
        #print('N. of files used: %s' %fname_list.shape[0])
        #print('n_classes: %s' %self.n_classes)
        #print('n_noisy_samples: %s' %self.n_noisy_samples)
        #print(self.batch_size//(self.n_noisy_samples))        
        #print(fname_list)
        for f_ind in range(len(fname_list)):            
            # Pick corresponding file from each folder 
              fname = fname_list[f_ind] 
              if self.Verbose:
                print('Loading file %s' %fname)
              #try:
              loaded_all = np.loadtxt(fname)
              P_original, k = loaded_all[:, 1:], loaded_all[:, 0]
              if self.sample_pace!=1:
                  P_original = P_original[0::self.sample_pace, :]
                  k = k[0::self.sample_pace]
                  P_original, k = P_original[:self.i_max], k[:self.i_max]
                  if self.Verbose:
                      print('Dimension of original data: %s' %str(P_original.shape))
                
              #except OSError:
              #  print('RUN WITH DIFFERENT SEED.')
              #  if len(self.list_IDs)==1:
              #      new_ind =  int((t_st.split('/')[-1]).split('.')[0])+len(self.labels)
                    #l = 
              #      fname1 = self.data_root + '/'+l+ '/'+ str(new_ind) + '.txt'
              #      print('File %s not found, trying with  %s'  %(fname, fname1))
                    
                    
              # Add noise
              for i_noise in range(self.n_noisy_samples):
                #i_ind=f_ind+i_noise
                if self.add_noise:
                  if self.Verbose:
                    print('Noise realization %s' %i_noise)
                  noise = np.random.normal(loc=0, scale=generate_noise(k,P_original, add_sys=self.add_sys,add_shot=self.add_shot,sigma_sys=self.sigma_sys ))
                  P_noisy = P_original+noise
                  expanded = np.expand_dims(P_noisy, axis=2)
                else:
                  if self.Verbose:
                    print('No noise')
                  expanded = np.expand_dims(P_original, axis=2) 
                # Store sample
                if self.Verbose:
                  print('Storing at position %s in the batch' %i_ind)
                  print('Dimension of data: %s' %str(expanded.shape))
                # swap axis if using one dim array in multiple channels 
                if self.swap_axes:
                    if self.Verbose:
                        print('Reshaping')
                    expanded = np.swapaxes(expanded, 2, 1)
                    if self.Verbose:
                        print('New dimension of data: %s' %str(expanded.shape))
                    expanded = expanded[:,:,self.z_bins]
                    if self.Verbose:
                        print('Final dimension of data: %s' %str(expanded.shape))
                    # now shape of expanded is (1, n_data_points, 1, n_channels=3)
                
                X[i_ind,] = expanded   
                
                                
                # Store class   
                label = fname.split('/')[-2]
                if not self.fine_tune:
                    encoding = self.labels_dict[label]
                else:
                    label = self.fine_tune_dict[label]
                    encoding = self.labels_dict[label]
                #if not self.fine_tune:
                #    ind = (f_ind)%(self.n_classes)
                #   assert label==self.inv_labels_dict[ind]
                
                if self.Verbose:
                  print('Label for this example: %s' %label)
                  print('Encoding: %s' % encoding)
                
                y[i_ind] = encoding
                i_ind += 1
        
        

        if self.normalization=='batch':
          mu_batch = np.mean(X, axis=0)
          std_batch = np.std(X, axis=0)
          X = (X-mu_batch)/std_batch
        elif self.normalization=='stdcosmo':
            if self.swap_axes:
                 X = X/self.norm_data[None, :, None, self.z_bins]-1
            else:
                X = X/self.norm_data[None, :,:,None]-1
        
        #if self.swap_axes:
        #    X = np.swapaxes(X, 3, 2)[:,:,:,1:]

        # shuffle to avoid having always three examples with different label in a row
        if self.shuffle:
          p = np.random.permutation(X.shape[0])
          X = X[p, :, :, :]
          y = y[p]
          if self.save_indexes:
            fname_list_shuffled = fname_list[p]
          
        if self.save_indexes:  
          if not os.path.exists(self.models_dir+'/idx_files/'):
              print('Creating directory %s' %self.models_dir+'/idx_files/')
              os.makedirs(self.models_dir+'/idx_files/')
            
          idx_file = self.models_dir+'/idx_files/idx_file_batch'+ str(self.batch_idx)+'.txt'                  
          #label_list = [f.split('/')[2] for f in fname_list]
          print('Saving indexes in  %s' %idx_file)
          idx_list = [int(f.split('.')[0].split('/')[-1] ) for f in fname_list_shuffled]
          self.save_indexes_dict[self.batch_idx] = idx_list
          
          with open(idx_file, 'w+') as file:
              print('Opened %s' %idx_file)
              for idx in idx_list: #i in range(len(idx_list)):
                  file.write("%i\n" %idx)
       

        #print('Seen in this batch: ')
        #print(list_IDs_temp)
    
        if self.swap_axes:# and self.z_bins.shape[0]>1:
            X = X[:,:,0,:] #np.squeeze(X)
        #elif self.swap_axes:
        #    X = X[:,:,0,:]
                
        
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    
    

    
    
def create_generators(FLAGS):
    
    
    os.chdir(FLAGS.my_path)
       
    
    # --------------------  CREATE DATA GENERATORS   --------------------
    
    
    all_index, n_samples, val_size, n_labels, labels, labels_dict = get_all_indexes(FLAGS)
    if FLAGS.fine_tune:
        n_labels_eff = n_labels*len(FLAGS.c_1)
    else:
        n_labels_eff=n_labels
    

    
    # SPLIT TRAIN/VALIDATION /(TEST)
    val_index = np.random.choice(all_index, size=int(np.floor(val_size*n_samples)), replace=False)
    train_index_temp =  np.setdiff1d(all_index, val_index) #np.delete(all_index, val_index-1)
    test_size_eff = FLAGS.test_size/(train_index_temp.shape[0]/n_samples)
    test_index = np.random.choice(train_index_temp, size=int(np.floor(test_size_eff*train_index_temp.shape[0])), replace=False)
    train_index =  np.setdiff1d(train_index_temp, test_index)

    print('Check for no duplicates in test: (0=ok):')
    print(np.array([np.isin(el, train_index) for el in test_index]).sum())
    print('Check for no duplicates in val: (0=ok):')
    print(np.array([np.isin(el, train_index) for el in val_index]).sum())

    print('N of files in training set: %s' %train_index.shape[0])
    print('N of files in validation set: %s' %val_index.shape[0])
    print('N of files in test set: %s' %test_index.shape[0])

    print('Check - total: %s' %(val_index.shape[0]+test_index.shape[0]+train_index.shape[0]))
    
    if FLAGS.add_noise:
        n_noisy_samples = FLAGS.n_noisy_samples
    else:
        n_noisy_samples = 1
    print('--Train')
    if FLAGS.test_mode:
        if not FLAGS.fine_tune:
            batch_size=train_index.shape[0]*n_labels_eff*n_noisy_samples
        else:
            batch_size=n_labels_eff*n_noisy_samples
    else:
        batch_size=FLAGS.batch_size
    print('batch_size: %s' %batch_size)

    if not FLAGS.test_mode:
        train_index_1  = cut_sample(train_index, batch_size, n_labels=n_labels_eff, n_noise=n_noisy_samples, Verbose=False)
        print('Train index length: %s' %train_index_1.shape[0])
    else:
        train_index_1 = train_index
        print('Train index: %s' %train_index_1)
    print('--Validation')
    if not FLAGS.test_mode:
        val_index_1  = cut_sample(val_index, batch_size, n_labels=n_labels_eff, n_noise=n_noisy_samples, Verbose=False)
        print('Val index length: %s'  %val_index_1.shape[0])
    else:
        val_index_1 = val_index
        print('Validation index: %s' %val_index_1)
    
    print('len(train_index_1), batch_size, n_labels_eff, n_noisy_samples = %s, %s, %s, %s' %(train_index_1.shape[0], batch_size, n_labels_eff,n_noisy_samples ))
    assert train_index_1.shape[0]%(batch_size//(n_labels_eff*n_noisy_samples))==0
    assert val_index_1.shape[0]%(batch_size//(n_labels_eff*n_noisy_samples))==0
    
    partition={'train': train_index_1, 'validation': val_index_1}
  
    ###################
    # USE THE BLOCH BELOW TO BE COMPATIBLE WITH OLDER VERSIONS OF DARTA GENERATORS. EVENTUALLY REMOVE
    ###################
    try:
        sigma_sys=FLAGS.sigma_sys
    except AttributeError:
        print(' ####  FLAGS.sigma_sys not found! #### \n Probably loading an older model. Using sigma_sys=0')
        sigma_sys=0.
        
    try:
        z_bins=FLAGS.z_bins
    except AttributeError:
        print(' ####  FLAGS.z_bins not found! #### \n Probably loading an older model. Using 4 z bins')
        z_bins=[0, 1, 2, 3]
    try:
        swap_axes=FLAGS.swap_axes
    except AttributeError:
        if FLAGS.im_channels>1:
            swap_axes=True
        else:
            swap_axes=False
        print(' ####  FLAGS.swap_axes not found! #### \n Probably loading an older model. Set swap_axes=%s' %str(swap_axes))
    ###################
    
    params = {'dim': (FLAGS.im_depth, FLAGS.im_width),
          'batch_size':batch_size, # should satisfy  m x Batch size /( n_labels x n_noisy_samples) =  n_indexes  with m a positive integer
          'n_channels': FLAGS.im_channels,
          'shuffle': True,
          'normalization': FLAGS.normalization,
          'sample_pace': FLAGS.sample_pace,
          'add_noise':FLAGS.add_noise,
          'n_noisy_samples':n_noisy_samples,
          'fine_tune':FLAGS.fine_tune,
          'add_shot':FLAGS.add_shot, 'add_sys':FLAGS.add_sys,
          'k_max':FLAGS.k_max, 'i_max':FLAGS.i_max, 'sigma_sys':sigma_sys,
          'swap_axes':swap_axes,
          'z_bins':z_bins
          }
    
    if FLAGS.fine_tune:
        params['c_0'] = FLAGS.c_0
        params['c_1'] = FLAGS.c_1
        params['fine_tune_dict'] = FLAGS.fine_tune_dict
        
    
    if not params['add_noise']:
        params['n_noisy_samples']=1
    
    print('--Train')
    training_generator = DataGenerator(partition['train'], labels, labels_dict, data_root = FLAGS.DIR, save_indexes=False, **params)
    print('--Validation')
    validation_generator = DataGenerator(partition['validation'], labels, labels_dict, data_root = FLAGS.DIR,  save_indexes=False, **params)

    
    
    return training_generator, validation_generator #, params




def create_test_generator(FLAGS):
    
    print('Changing directory to %s' %FLAGS.my_path)
    os.chdir(FLAGS.my_path)
    
    all_index, n_samples, val_size, n_labels, labels, labels_dict = get_all_indexes(FLAGS, Test=True)
    
    if FLAGS.fine_tune:
        n_labels_eff = n_labels*len(FLAGS.c_1)
    else:
        n_labels_eff = n_labels
    
    if FLAGS.add_noise:
        n_noisy_samples = FLAGS.n_noisy_samples
    else:
        n_noisy_samples = 1
    print('--Train')
    if FLAGS.test_mode:
        if not FLAGS.fine_tune:
            batch_size=all_index.shape[0]*n_labels_eff*n_noisy_samples
        else:
            batch_size=n_labels_eff*n_noisy_samples
    else:
        batch_size=FLAGS.batch_size
    print('batch_size: %s' %batch_size)
        
    test_index_1  = cut_sample(all_index, batch_size, n_labels=n_labels_eff, n_noise=n_noisy_samples, Verbose=True)
    n_test = test_index_1.shape[0]

    assert test_index_1.shape[0]%(batch_size//(n_labels_eff*n_noisy_samples))==0

    print('N. of test files used: %s' %n_test)

    partition_test = {'test': test_index_1}
    
    
 
    
    params_test = {'dim': (FLAGS.im_depth, FLAGS.im_width),
          'batch_size':batch_size, # should satisfy  m x Batch size /( n_labels x n_noisy_samples) =  n_indexes  with m a positive integer
          'n_channels': FLAGS.im_channels,
          'shuffle': True,
          'normalization': FLAGS.normalization,
          'sample_pace': FLAGS.sample_pace,
          'add_noise':FLAGS.add_noise,
          'n_noisy_samples':n_noisy_samples, 
          'fine_tune':FLAGS.fine_tune,
          'add_shot':FLAGS.add_shot, 'add_sys':FLAGS.add_sys,
          'k_max':FLAGS.k_max, 'i_max':FLAGS.i_max, 'sigma_sys':FLAGS.sigma_sys,
          'swap_axes':FLAGS.swap_axes,
          'z_bins':FLAGS.z_bins,
          }
    
    if FLAGS.fine_tune:
        params_test['c_0'] = FLAGS.c_0
        params_test['c_1'] = FLAGS.c_1
        params_test['fine_tune_dict'] = FLAGS.fine_tune_dict
        

    if not params_test['add_noise']:
        params_test['n_noisy_samples']=1
    
    
    test_generator = DataGenerator(partition_test['test'], 
                                   labels, labels_dict, 
                                   data_root=FLAGS.TEST_DIR , 
                               save_indexes = FLAGS.save_indexes,
                               models_dir=FLAGS.models_dir+FLAGS.fname,
                               idx_file_name = FLAGS.fname, 
                               **params_test)

    
    
    return test_generator