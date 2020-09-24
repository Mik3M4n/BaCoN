#!/bin/bash
echo "Initializing training data"
echo ""

cd "$(dirname "$0")"

#need to figure out how to just use integers in commands .... -.-
x=1;
xx=2;
y=3;
yy=4;
z=5;
zz=6;
zzz=7;

#basic quantities
zvals=`awk 'NR=='${x}' {print $3}' configurations.dat`
kmin=`awk 'NR=='${xx}' {print $3}' configurations.dat`
kmax=`awk 'NR=='${y}' {print $3}' configurations.dat`
NK=`awk 'NR=='${yy}' {print $3}' configurations.dat`
name=`awk 'NR=='${z}' {print $3}' configurations.dat`
cambdir=`awk 'NR=='${zz}' {print $3}' configurations.dat`
copdir=`awk 'NR=='${zzz}' {print $3}' configurations.dat`


# Start the loop over parameter values from myfile.txt
for iteration in $(seq 1 1 2500)
do

cp copter.ini_template copter.ini

#iteration=1;

echo "Creating training data number: ${iteration}"
echo ""


#Omega_m,  Omega_b, omega_cdm, hubble2, n_s
param1=`awk 'NR=='${iteration}' {print $2}' myfile.txt`
param2=`awk 'NR=='${iteration}' {print $3}' myfile.txt`
param1a=`awk 'NR=='${iteration}' {print $2-'${param2}'}' myfile.txt`
param3=`awk 'NR=='${iteration}' {print $4}' myfile.txt`
param4=`awk 'NR=='${iteration}' {print $5}' myfile.txt`

#small h
hnorm=100;
param3a=`awk 'NR=='${iteration}' {print $4/'${hnorm}'}' myfile.txt`
#Omega_lambda
ol=`awk 'NR=='${iteration}' {print '${x}'-$2}' myfile.txt`

# sigma8
param5=`awk 'NR=='${iteration}' {print $6}' myfile.txt`
#mg params
param6=`awk 'NR=='${iteration}' {print $7}' myfile.txt`
param7=`awk 'NR=='${iteration}' {print $8}' myfile.txt`
param8=`awk 'NR=='${iteration}' {print $9}' myfile.txt`


#edit camb .ini file with relevant params (sigma8 and fr0 are given to copter directly)
sed -i "" "s/omega_cdm =/omega_cdm = ${param1a}/g" copter.ini
sed -i "" "s/omega_baryon =/omega_baryon = ${param2}/g" copter.ini
sed -i "" "s/hubble =/hubble = ${param3}/g" copter.ini
sed -i "" "s/scalar_spectral_index(1) =/scalar_spectral_index(1) = ${param4}/g" copter.ini
sed -i "" "s/omega_lambda =/omega_lambda = ${ol}/g" copter.ini

#name2=`awk 'NR=='${q}' {print $3 '_''${iteration}'}' configurations.dat`

mv copter.ini ${cambdir}

cd ${cambdir}

echo "Running CAMB with parameters:"
echo ""
echo "h='${param3a}'"
echo "ns='${param4}'"
echo "Omega_b='${param2}'"
echo "Omega_m='${param1}'"
echo "s8='${param5}'"
echo "Omega_L='${ol}'"
echo "Name of file='${name}'"
echo "Modified grav param 1 ='${param6}'"
echo "Modified grav param 2 ='${param7}'"
echo "Modified grav param 3 ='${param8}'"
echo "Computed at redshifts ='${zvals}'"
echo "Number of k values ='${NK}'"
echo ""

./camb copter.ini #> /dev/null

norm=`awk 'NR=='${x}' {print $7}' test_transfer_out.dat`

awk '{print $1,$7/'${norm}'}' test_transfer_out.dat | column -t > ${name}.dat

mv ${name}.dat ${copdir}/examples/transfers

cd ${copdir}/examples/transfers

cp myini.ini_template ${name}.ini

echo "Creating .ini file ... "
echo ""

sed -i "" "s/h =/h = ${param3a}/g" ${name}.ini
sed -i "" "s/n =/n = ${param4}/g" ${name}.ini
sed -i "" "s/Omega_b =/Omega_b = ${param2}/g" ${name}.ini
sed -i "" "s/Omega_m =/Omega_m = ${param1}/g" ${name}.ini
sed -i "" "s/sigma8 =/sigma8 = ${param5}/g" ${name}.ini
sed -i "" "s/tkfile =/tkfile = transfers\/${name}.dat/g" ${name}.ini


echo "Editing react_ml.cpp ... "
echo ""

cd ${copdir}/examples

cp react_ml.cpp_template react_ml.cpp


sed -i "" "s/const char\* cstr =/const char\* cstr = \"transfers\/${name}\";/g" react_ml.cpp
sed -i "" "s/const char\* output =/const char\* output = \"data\/${iteration}.txt\";/g" react_ml.cpp

sed -i "" "s/double kmax =/double kmax = ${kmax};/g" react_ml.cpp
sed -i "" "s/double kmin =/double kmin = ${kmin};/g" react_ml.cpp
sed -i "" "s/int Nk =/int Nk = ${NK};/g" react_ml.cpp

sed -i "" "s/double redshifts\[Nz\] =/double redshifts\[Nz\] = ${zvals};/g" react_ml.cpp
sed -i "" "s/vars\[1\] =/vars\[1\] = ${param1};/g" react_ml.cpp
sed -i "" "s/vars\[2\] =/vars\[2\] = ${param6};/g" react_ml.cpp
sed -i "" "s/vars\[3\] =/vars\[3\] = ${param7};/g" react_ml.cpp
sed -i "" "s/vars\[4\] =/vars\[4\] = ${param8};/g" react_ml.cpp

echo "Running COPTER ... "
echo ""

g++ -I/${copdir}/include -lgsl -lstdc++ -L/${copdir}/lib -lcopter react_ml.cpp -o test

time ./test

echo "DONE!!! :D "

done
