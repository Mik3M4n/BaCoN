# BaCoN (BAyesian COsmological Network)
This package allows to train and test Bayesian Convolutional Neural Networks in order to **classify dark matter power spectra as being representative of different cosmologies**, as well as to compute the classification confidence. 
The code now supports the following theories:  **LCDM, wCDM, f(R), DGP, and a randomly generated class** (see the reference for details). Additional cosmologies can be easily added (see the dedicated section).

**We also provide a jupyter notebook that allows to load the pre-trained model, classify any matter power spectrum and compute the classification confidence with the method described in the paper (see [4 - Classification](https://github.com/Mik3M4n/BaCoN#4---Classification)). This only requires the raw values of the power spectrum. Feedback on the results of classification is particularly welcome!**

If you have a code that generates power spectra of modified gravity theories up to k>=2.5 h Mpc^_1, get in touch ;)

The methods and results can be found in the paper [Seeking New Physics in Cosmology with Bayesian Neural Networks I: Dark Energy and Modified Gravity](). 

Please contact <Michele.Mancarella@unige.ch> for further questions.


## Summary


* [Citation](https://github.com/Mik3M4n/BaCoN#citation)
* [Overview and code organisation](https://github.com/Mik3M4n/BaCoN#Overview and code organisation)
* [Data](https://github.com/Mik3M4n/BaCoN#Data)
	* [Data folders](https://github.com/Mik3M4n/BaCoN#Data-folders)
	*  [Data format](https://github.com/Mik3M4n/BaCoN#Data-format)
	*  [Data generator](https://github.com/Mik3M4n/BaCoN#Data-generator)
* [Usage](https://github.com/Mik3M4n/BaCoN#Usage)
	* [1a - Training five-label networks](https://github.com/Mik3M4n/BaCoN#1a---Training-five-label-networks)
	* [1b - Training two-label networks](https://github.com/Mik3M4n/BaCoN#1b---Training-two-label-networks)
	* [1c - Training specialist networks](https://github.com/Mik3M4n/BaCoN#1c---Training-specialist-networks)
	* [1d - Training custom networks](https://github.com/Mik3M4n/BaCoN#1d---Training-custom-networks)
	* [Output](https://github.com/Mik3M4n/BaCoN#Output)
	* [2 - Fine-tuning](https://github.com/Mik3M4n/BaCoN#2---Fine-tuning)
	* [3 - Testing](https://github.com/Mik3M4n/BaCoN#3---Testing)
	* [4 - Classification](https://github.com/Mik3M4n/BaCoN#4---Classification)
* [Adding cosmologies](https://github.com/Mik3M4n/BaCoN#Adding-cosmologies)
* [Modifying the code](https://github.com/Mik3M4n/BaCoN#Modifying-the-code)

## Citation
This package is released together with the paper [Seeking New Physics in Cosmology with Bayesian Neural Networks I: Dark Energy and Modified Gravity](). When making use of it, please cite the paper and the present git repository. Bibtex:



## Overview and code organisation


The package provides the following modules:

* ```data generator.py```: data generator that generates batches of data. Data are dark matter power spectra normalised to the Planck LCDM cosmology, in the redshift bins (0.1,0.478,0.783,1.5) and k in the range 0.01-2.5 h Mpc^-1.
* ```models.py``` : contains models' architecture
* ```train.py```: module to train and fine-tune models. Checkpoints are automatically saved each time the validation loss decreases. Both bayesian and "traditional" NNs are available.
* ```test.py```: evaluates the accuracy and the confusion matrix.

A jupyter notebook to classify power spectra with pre-trained weights and computing the confidence in classification is available in ```notebooks/```. Pre-trained models are available in ```models/```.

Furthermore, we provide scripts to generate the "random" class spectra with the algorithm described in the paper, as well as scripts to generate datasets with the publicly available code [```ReACT```](https://github.com/nebblu/ReACT) in the folder ```scripts/```.

We include two pre-trained models.

The first base model is a five-label classifier with LCDM, wCDM, f(R), DGP, and "random" as classes, while the second is a two-label classifier with classes LCDM and non-LCDM.

On top of the five-labels classifier, a fine-tuning procedure is also available, in order to adapt the five-label network to LCDM and non-LCDM classes only. This can be useful in the search for new physics, irrespectively of the underlying alternative model, if one wants to add theories that are not included in the five labels network without re-training from scratch. The fine-tuning starts from a model pre-trained on the 5 labels task, replacing the final, 5-dimensional output with a two dimensional output layer.

Details on training, data preparation, variations ot the base model, and extensions are available in the dedicated sections. The Bayesian implementation uses [Tensorflow probability](https://www.tensorflow.org/probability) with [Convolutional](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution1DFlipout) and [DenseFlipout](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout) methods.



## Data

### Data folders

The data used to train this network are available at [this link](https://doi.org/10.5281/zenodo.4309918). 
They should be downloaded in the ```data/``` folder. Its organisation should look as follows:
```bash
data/train_data/
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
		├── wcdm/
			├──1.txt
			├──2.txt
			├──...	
		└── planck.txt		
```
and similarly for the test data that should be put in a directory ```test_data``` inside ```data/```.
The file names in each folder must have the same indexing, ranging from 1 to the number of training/test examples for each label. Note that despite the names, data in each folder should be uncorrelated. Nonetheless, the data generator shuffles the indexes so that different indexes in different folders are used when producing a batch, in order to further prevent any spurious correlation that may arise when generating data.
Scripts to generate the data as well as to correctly re-index are available in ```scripts```.
The data generator will automatically check the data folder and assign labels corresponding to the subfolders' names. At test time, make sure that the test data folder has the same names as the training set in order not to run into errors.
The file ```planck.txt``` contains the reference power spectrum used to normalize the input to the network. We provide it inside the data and in the folder data/example_spectra/ in this repository in order to use the examples.

### Data format

Each file should contain in the first column the values of k and in each of the following columns the values of P(k) at different redshift bins ordered from the highest to the lowest redshift. The data are generated in redshift bins (0.1,0.478,0.783,1.5) [If generating new data with different redshifst bins, note that the noise in Eq. 3.1 makes use of the galaxy number density and comoving volume at the corresponding z, so this should be changed in ```data_generator.py``` if using different redshift bins].

The provided data contain 500 redshift bins between 0.01 and 10 h Mpc^-1. In order to down-sample to have a number of k bins comparable to Euclid, we sample one every four points. This is done with the option ```sample_pace=4``` (default) in ```DataGenerator ``` (see next section) or when using ```train.py``` (see *Usage* ). Furthermore, we restrict to k<2.5 h Mpc^_1 . This can be done in two ways: either by specifying ```k_max=2.5``` (default) in ```DataGenerator ```/```train.py```, in which case the code will restrict to k<k\_max, or specifying ```i_max```, i.e. the index suck that all k<k[i_max] are <k\_max. (Note that this cut is applied after the sampling). 

The default values are ```sample_pace=4```, ```k_max=2.5```.

### Data generator

The ```DataGenerator``` object contained in ```data generator.py``` generates batches of data. Each batch consists of power spectra with Euclid-like noise, normalized to a reference LCDM spectrum with Planck2015 cosmology. This spectrum is in a file ```planck.txt``` that has to be inside the main data directory. Otherwise, the path to this file can be specified to the ```norm_data_name``` option. 
The basic data generator can be initialized with the following syntax (a full list of options is available below):

```
train_generator = DataGenerator(list_IDs, labels, labels_dict, batch_size=batch_size)
```
where

* ```list_IDs ``` is a list of the indexes that should be used to choose from each folder the files to use in a training set. 
* ```labels``` is a list of the labels. For example, for the five-labels network, ```labels=['dgp', 'fR', 'lcdm', 'rand', 'wcdm']```
* ```labels_dict ``` is a dictionary of the form ```{label: integer}``` specifying the labels encoding. For example, for the five-labels network, ```labels_dict={'dgp': 0, 'fR': 1, 'lcdm': 2, 'rand': 3, 'wcdm': 4} ```
* ```batch_size``` is the dimension of the batch. Note that each batch is composed by an equal number of spectra in each class. Noise is then added to each example, and one can have multiple noise realizations for each underlying spectrum. By default, in each batch a single spectrum is added with 10 different noise realizations. Hence, the batch size must be multiple of the number of labels times the number of noise realizations. For example, for the five-labels network with 10 noise realizations, the batch size must be multiple of 5x10, which is also the minimum batch size possible.
 
When generating batches of data, the data generator performs the following tasks:

* Selects among ```list_IDs ``` the subset of indexes used for the given batch for each of the classes.
* For each index and class, loads the corresponding spectrum.
* Adds different realizations of noise to each spectrum. The noise is gaussian with std given by Eq. 3.1 of the paper
* Normalizes by the reference spectrum
* Shuffles the batch in order not to have all examples coming from the same underlying theory in a row

For example, a batch of size 50 for the five-labels network will be built using one index per each class, thus loading 5 different power spectra one for each theory, then adding to each of them 10 realizations of the noise and normalizing.

The labels are returned in one-hot-encoding, ordered in alphabetical order.

## Usage

### 1a - Training five-label networks

To train a five-label model as in the paper, and save the result in ```models/my_model/```:

```
python train.py --fname='my_model'
```

Note that if running on Google Colab, the saving of the log file contained in the code sometime fails. In this case the output has to be manually redirected to file. This can be done with ```tee```:

```
python train.py --fname='my_model' | tee my_model_log.txt; mv my_model_log.txt my_model/my_model
```

It is recommended to run in test mode before running a complete training, to check that the output is correctly saved. This means that the network trains only on one batch of the minimal size. This is done by adding ```test_mode='True'```.

### 1b - Training two-label networks

A two-label network is trained on two classes, LCDM and non-LCDM. At training time, the option ```one_vs_all``` must be set to ```True```:

```
python train.py --fname='my_model' --one_vs_all='True'
```

One can select which theories to include in the non-LCDM class (by default, all the non-LCDM theories present in the dataset are included). This is done by passing a list of the labels to include in the non-LCDM class through the argument ```c_1```. For example, to include only fR and DGP:

```
python train.py --fname='my_model' --one_vs_all='True' --c_1 'fR' 'dgp'
```


Moreover, there are multiple options to build a batch of data:

* ```dataset_balanced='False'``` : each batch is composed by one LCDM example and one example for each of the non-LCDM classes (default)
* ```dataset_balanced='True'``` : each batch is composed by half examples from LCDM and the remaining half equally split between the non-LCDM classes 

Note that the batch size must be chosen with the correct dimension in order to respect the above proportions, or the code will throw an error.

We find that the first option performs better.

### 1c - Training specialist networks

This is a subcase of a two-label network, where the non-LCDM class consists of a single theory. This is achieved by passing the corresponding label through the argument ```c_1```, e.g. for a specialist fR network: ```--c_1 'fR'```

### 1d - Training custom networks
The code allows to vary the number of convolutional and dense layers, filter size, stride, padding, kernel size, batch normalization and pooling layers, number of z bins and max k.

Full list of options (more detailed doc coming soon...):

```
train.py [-h] [--bayesian BAYESIAN] [--test_mode TEST_MODE]
                [--n_test_idx N_TEST_IDX] [--seed SEED]
                [--fine_tune FINE_TUNE] [--one_vs_all ONE_VS_ALL]
                [--c_0 C_0 [C_0 ...]] [--c_1 C_1 [C_1 ...]]
                [--dataset_balanced DATASET_BALANCED]
                [--include_last INCLUDE_LAST] [--log_path LOG_PATH]
                [--restore RESTORE] [--fname FNAME] [--model_name MODEL_NAME]
                [--my_path MY_PATH] [--DIR DIR] [--TEST_DIR TEST_DIR]
                [--models_dir MODELS_DIR] [--save_ckpt SAVE_CKPT]
                [--out_path_overwrite OUT_PATH_OVERWRITE]
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
                [--add_FT_dense ADD_FT_DENSE] [--trainable TRAINABLE]
                [--unfreeze UNFREEZE] [--lr LR] [--drop DROP]
                [--n_epochs N_EPOCHS] [--val_size VAL_SIZE]
                [--test_size TEST_SIZE] [--batch_size BATCH_SIZE]
                [--patience PATIENCE] [--GPU GPU] [--decay DECAY]
                [--BatchNorm BATCHNORM]
```

### Output
The results will be saved in a new folder inside the folder specified in the input parameter ```mdir```. The folder name is passed to the code through the parameter ```fname```. 
At the end of training, the ouput folder will contain (see ```models/``` for an example):

* a plot of the learning curves (training and validation accuracies and losses) in ```hist.png```.
*  a subfolder tf_ckpts containing:
	* 	 the checkpoints of the models 
	* 	  four ```.txt``` files with the history of test and validation accuracies and losses, namely:
		* hist_accuracy.txt : training accuracy
		* hist_val\_accuracies.txt: validation accuracy
		* hist_loss.txt: training loss
		* hist_val\_loss.txt: validation loss
	* two files ```idxs_train.txt``` and ```idxs_val.txt``` containing the indexes of the files used for the training and validaiton sets, to make the training set reproducible (note however that noise realizations will add randomness to the underlying spectra.).

### 2 - Fine-tuning

The module is ```train.py```, but one has to set the option ```fine_tune``` to ```True``` and to specify the path to the fine tuning data folder. This is done throught the parameter ```DIR```.
The path to the log file of the original training must be specified, in order for the training module to load the correct options and model architecture. Otherwise, these muse be specified manually.

Being this a two-label network, the same options as for the two-label case can be chosen for building a batch (See 1b).

By default, the final, 5-d layer is dropped and replaced by a 2-d output layer.

Additional parameters for fine tuning: 

* ```add_FT_dense```: if ```True```, adds an additional dense layer before the final 2D one
* ```trainable```: if ```True```, makes the full network trainable, otherwise only the last layer, leaving the rest frozen. Default is ```False```.
* ```Unfreeze```: if ```True```, allows to additionally fully re-train a fine-tuned network where at fine tuning time ```trainable=False```. Default is ```False```

For example, to fine-tune the five-labels network for 10 epochs, with data contained in ```data/fine_tuning_data/```:

```
python train.py --fine_tune='True' --one_vs_all='True' --log_path="models/five_label/five_label_log.txt" --fname='five_label' --DIR='data/fine_tuning_data/' --n_epochs=10  
```

### 3 - Testing
The ```test.py``` module loads the weights of a previoulsy trained model, computes the accuracy on the test data and outputs a confusion matrix. The result will be saved in the same folder as the training.

For a bayesian net, classification is performed by drawing MC samples from the weights and averaging the result. The number of samples (default 100) can be specified via ```n_monte_carlo_samples ```. One example is classified if the max probability among the labels exceeds a threshold values of 0.5 . This threshold can be changed with the argument ```th_prob ```. In the confiusion matrix, a "Non Classified" class is added to account for this examples. THe accuracy is also computed leaing the un classified examples out, for comparison.

One can also vary the number of noise realizations and noise options at test time.
Finally, if using only one noise realization ```n_noisy_samples =1```, there is the possibility of saving the indexes of the spectra used in each batch, in order to be able to check the corresponding parameters' values. This is done by setting ```save_indexes='True'```.

To test the five-label network:

```
python test.py --log_path='models/five_label/five_label_log.txt'
```


Full options:

```
test.py [-h] --log_path LOG_PATH [--TEST_DIR TEST_DIR]
               [--models_dir MODELS_DIR]
               [--n_monte_carlo_samples N_MONTE_CARLO_SAMPLES]
               [--th_prob TH_PROB] [--batch_size BATCH_SIZE]
               [--add_noise ADD_NOISE] [--n_noisy_samples N_NOISY_SAMPLES]
               [--add_shot ADD_SHOT] [--add_sys ADD_SYS]
               [--sigma_sys SIGMA_SYS] [--save_indexes SAVE_INDEXES]
```



### 4 - Classification

We provide an example of classification in the notebook ```Classification.ipynb``` in ```notebooks/```. This implements the computation of the confidence as described in Sec. 2.3 and Appendix B of the paper, and can be used to reproduce Fig. 6-8-9.

One can also classify any new power spectrum and compute the confidence in the classification. This is done in a straightforward way: it is sufficient to save the data in the correct format (see *Data*) and to follow the steps outlined in the notebook. 

The notebook can be run directly in Google Colab.


## Adding cosmologies

It is sufficient to add new dark matter PS to the ```train_data``` folder, in a subfolder with the name of the theory. The data should be in the same format as the others (see *Data*) and with the correct indexing. Nothing else is needed to train. The code automatically assigns as labels the names of the data subfolders, with encoding in alphabetical order.

Happy training! 

## Modifying the code

Coming soon ...