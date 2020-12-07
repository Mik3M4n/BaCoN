# BaCoN (BAyesian COsmological Network)
This package allows to train and test a Bayesian Convolutional Neural Network in order to classify dark matter power spectra as being representative of different cosmologies, as well as to compute the classification confidence as described in [](). 

The Bayesian implementation uses [Tensorflow probability](https://www.tensorflow.org/probability) with [Convolutional](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution1DFlipout) and [DenseFlipout](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout) methods.

The two base models are trained on five different categories: LCDM, wCDM, f(R), DGP, and a randomly generated spectrum (see the reference for details). Additional cosmologies can be easily added (see the dedicated section). The first base model is a five-label classifier with LCDM, wCDM, f(R), DGP, and "random" as classes, while the second is a two-label classifier with classes LCDM and non-LCDM.

On top of the five-labels classifier, a fine-tuning procedure is also available, in order to adapt the five-label network to LCDM and non-LCDM classes only. This can be useful in the search for new physics, irrespectively of the underlying alternative model, if one wants to add theories that are not included in the five labels network without re-training from scratch. The fine-tuning starts from a model pre-trained on the 5 labels task, replacing the final, 5-dimensional output with a two dimensional output layer.

Details on training, data preparation, variations ot the base model, and extensions are available in the dedicated sections.

We also provide a jupyter notebook that allows to load the pre-trained model, classify any matter power spectrum and compute the classification confidence with the method described in the paper. 

Please also refer to the paper for details.


## Citation
This package is released together with the paper [Seeking New Physics in Cosmology with Bayesian Neural Networks I: Dark Energy and Modified Gravity](). When making use of it, please cite the paper and the present git repository. Bibtex:




## Overview and code organisation


This package provides the following modules:

* ```data generator.py```: data generator that generates batches of data. Data are dark matter power spectra normalised to the Planck LCDM cosmology, in the redshift bins (0.1,0.478,0.783,1.5) and k in the range 0.01-2.5 h Mpc^-1.
* ```models.py``` : contains models' architecture
* ```train.py```: module to train and fine-tune models. Checkpoints are automatically saved each time the validation loss decreases. Both bayesian and "traditional" NNs are available.
* ```test.py```: evaluates the accuracy and the confusion matrix.

A jupyter notebook to classify power spectra with pre-trained weights and computing the confidence in classification is available in ```notebooks/```

Furthermore, we provide scripts to generate the "random" class spectra with the algorithm described in the paper, as well as scripts to generate datasets with the publicly available code [```ReACT```](https://github.com/nebblu/ReACT) in the folder ```scripts```.

## Data
Data should be put inside the main repository folder. Its organisation should be as follows:
```bash
train_data/
	├── dgp/
			├──1.txt
			├──2.txt
			├──...
	├── fR/
			├──1.txt
			├──2.txt
			├──...
	├── lcdm/
			├──1.txt
			├──2.txt
			├──...
	├── rand/
			├──1.txt
			├──2.txt
			├──...	
	└── wcdm/
			├──1.txt
			├──2.txt
			├──...			
```
The file names in each folder must have the same indexing. Scripts to generate the data as well as to correctly re-index are available in ```scripts```.
The data generator will automatically check the data folder and assign labels corresponding to the subfolders' names. At test time, make sure that the test data folder has the same names as the training set in order not to run into errors.

## Usage

### 1 - Training
```
train.py [-h] [--bayesian BAYESIAN] [--test_mode TEST_MODE]
                [--n_test_idx N_TEST_IDX] [--seed SEED]
                [--fine_tune FINE_TUNE] [--log_path LOG_PATH]
                [--restore RESTORE] [--fname FNAME] [--model_name MODEL_NAME]
                [--my_path MY_PATH] [--DIR DIR] [--TEST_DIR TEST_DIR]
                [--models_dir MODELS_DIR] [--save_ckpt SAVE_CKPT]
                [--im_depth IM_DEPTH] [--im_width IM_WIDTH]
                [--im_channels IM_CHANNELS] [--swap_axes SWAP_AXES]
                [--sort_labels SORT_LABELS] [--normalization NORMALIZATION]
                [--sample_pace SAMPLE_PACE] [--k_max K_MAX] [--i_max I_MAX]
                [--add_noise ADD_NOISE] [--n_noisy_samples N_NOISY_SAMPLES]
                [--add_shot ADD_SHOT] [--add_sys ADD_SYS]
                [--sigma_sys SIGMA_SYS] [--z_bins Z_BINS [Z_BINS ...]]
                [--n_dense N_DENSE] [--filters FILTERS [FILTERS ...]]
                [--kernel_sizes KERNEL_SIZES [KERNEL_SIZES ...]]
                [--strides STRIDES [STRIDES ...]]
                [--pool_sizes POOL_SIZES [POOL_SIZES ...]]
                [--strides_pooling STRIDES_POOLING [STRIDES_POOLING ...]]
                [--add_FT_dense ADD_FT_DENSE] [--lr LR] [--drop DROP]
                [--n_epochs N_EPOCHS] [--val_size VAL_SIZE]
                [--test_size TEST_SIZE] [--batch_size BATCH_SIZE]
                [--patience PATIENCE] [--GPU GPU] [--decay DECAY]
```
#### Output
The results will be saved in a new folder inside the folder specified in the input parameter mdir. The folder name is passed to the code with the parameter fname. 
At the end of training, the ouput folder will contain:

* a plot of the learning curves (training and validation accuracies and losses) 
*  a subfolder tf_ckpts containing:
	* 	 the checkpoints of the models 
	* 	  four .txt files with the history of test and validation accuracies and losses, namely:
		* hist_accuracy.txt : training accuracy
		* hist_val\_accuracies.txt: validation accuracy
		* hist_loss.txt: training loss
		* hist_val\_loss.txt: validation loss

### 2 - Fine-tuning

The module is ```train.py```, but one has to set the option ```fine_tune``` to ```True```. 


### 3 - Testing
```
test.py [-h] --log_path LOG_PATH [--TEST_DIR TEST_DIR]
               [--n_monte_carlo_samples N_MONTE_CARLO_SAMPLES]
               [--th_prob TH_PROB] [--batch_size BATCH_SIZE]
               [--add_noise ADD_NOISE] [--n_noisy_samples N_NOISY_SAMPLES]
               [--add_shot ADD_SHOT] [--add_sys ADD_SYS]
               [--sigma_sys SIGMA_SYS]
```

#### Output

### Classification

## Examples


## Adding cosmologies

## Modifying the code