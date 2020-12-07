# Random filter generation. 

Scripts used to generate the random filters as described in Appendix A. 

1) Use *random_pk/Filter.nb* to generate filters. 

2) Place reference spectra in *random\_pk/ref\_pk* in class labeled folders. 

3) Compile and run *random\_pk\_gen.cpp* to apply filters to the reference spectra: 

> g++ -lgsl random\_pk\_gen.cpp 
> ./a.out 

Make sure the number of reference pk (noref) and number of filters (Nfilters) is consistent between *random\_pk\_gen.cpp* and those included in the *random\_pk* folder.
