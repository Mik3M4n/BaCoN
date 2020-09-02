# Seeking New Physics in Cosmology with Bayesian Neural Networks I: Dark Energy and Modified Gravity

This package allows to train and test a Bayesian Convolutional Neural Network in order to classify matter power spectra as being representative of different cosmologies. The base model is trained on five different categories: LCDM, wCDM, f(R), DGP, and a randomly generated spectrum (see the paper for details). Additional cosmologies can be easily added (see the dedicated section). On top of this 5-labels classifiers, a fine-tuning procedure is available in order to classify a power spectrum as belonging to the LCDM class or not. This is useful in the search for new physics, irrespectively of the underlying alternative model. The fine-tuning starts from a model pre-trained on the 5 labels task, replacing the final, 5-dimensional output with a two dimensional output layer.

**When making use of this package, please cite [this paper]() and the present git repository.**


## Overview and code organisation


This package provides:

* A data generator that generates batches of data. Data are normalised to the Planck LCDM cosmology
* A train module. A fine tuning option is available to train the model on the binary task LCDM-non LCDM. Checkpoints are automatically saved each time the validation loss decreases.
* A test module. This evaluates the accuracy and a confusion matrix. In the bayesian case, this outputs a histogral of the accuracies obtained by passing a test example through the network multiple times.


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

## Examples


## Adding cosmologies

## Modifying the code