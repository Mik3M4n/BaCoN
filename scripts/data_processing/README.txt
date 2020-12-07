# Scripts used to process output data 

These scripts are used to perform some data processing on the output power spectra produced by ReACT. A brief description of these are given below: 

check_indices : checks what missing indices are in a specified folder 

delete : deletes index files specified by an array in a specified folder  

relabel : relabels index files in specified folder from one set of indices to another specified as arrays.

rename : shifts index files in a specified folder. 

relabel_params.cpp : relabels the indices of a parameter value file (dgp_params.txt default) to those specified in an array. 

> g++ -lgsl relabel_params.cpp 
>./a.out 
