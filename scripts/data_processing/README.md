# Scripts used to process output data 

These scripts are used to perform some additional data processing on the output power spectra produced by ReACT. A brief description of these are given below. For more details please open the individual files. 

**check_indices.sh**: checks what missing indices are in a specified data folder, where the data files inside the specified folder are labeled as *index*.txt. 

**delete.sh** : deletes index files specified by an array in a specified folder  

**relabel.sh** : relabels index files in specified folder from one set of indices to another specified as arrays. These indices are specified within the file. 

**rename.sh** : shifts index files in a specified folder. 

**relabel_params.cpp** : relabels the indices of a parameter value file (*dgp_params.txt* is the default target parameter file) to those specified in an array. See the data_generation_react folder README for more details on the parameter file format.  

> g++ -lgsl relabel_params.cpp 
>./a.out 
