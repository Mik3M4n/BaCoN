#!/bin/bash
echo "Initializing training data"
echo ""

cd "$(dirname "$0")"

cd /Users/bbose/Desktop/ML+ReACT/reactions/examples/lcdm
mynum=5000;

# Start the loop over parameter values from myfile.txt
for iteration in $(seq 1 1 5000)
do

newname=$((iteration+mynum));

echo "This is the new name ${newname}"

mv  ${iteration}.txt ${newname}.txt

echo "Renaming training data number: ${iteration}"

echo "DONE!!! :D "

done
