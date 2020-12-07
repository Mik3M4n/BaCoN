# Scripts used to generate data with [ReACT](https://github.com/nebblu/ReACT)

These scripts are used to generate the power spectrum data files used in training and testing BaCoN. 

1) Place all the files in the reactions/examples  folder, except copter.ini_template, which should go in reactions/examples/transfers. 

2) Modify the configuration.dat file to include your CAMB and ReACT directories as well as the redshifts, k-bins and k-ranges you want to output at. k-modes are sampled logarithmically. Redshifts should be specified in descending order. 

3) Generate a cosmology/gravity parameter file with param_generation.nb, or just create your own file with the format: 

Index Omegam Omegab H0 ns  sigma8* param1 param2 param3 

Where param1-3 are the additional parameters of the model. Ex. Param1 = fr0 for f(R). These should correspond to the parametrisation included in reactions/src/SpecialFunctions.cpp. 

4) Run:


>./run_react.sh 

# Note on sigma_8*: 

The value specified in the myfile.txt is not the true sigma_8 but a rescaling. You can find the ‘true’ LCDM value of sigma_8 by:   

> sigma_8(true) = ( sigma_8(camb)  / sigma_8* ) x sigma_8(camb)

Where sigma_8(camb) is the value given by CAMB.  