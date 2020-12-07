#!/bin/bash
echo "Initializing training data"
echo ""

cd "$(dirname "$0")"

# Specify directory with files to be renamed by a shift in index - here by 5000
cd /Users/bbose/Desktop/ML+ReACT/reactions/examples/lcdm
mynum=5000;

for iteration in $(seq 1 1 5000)
do

newname=$((iteration+mynum));

echo "This is the new name ${newname}"

mv  ${iteration}.txt ${newname}.txt

echo "Renaming training data number: ${iteration}"

echo "DONE!"

done
