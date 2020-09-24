#!/bin/bash
echo "Starting ...."
echo ""

cd "$(dirname "$0")"

cd /Users/bbose/Desktop/ML+ReACT/reactions/examples/randpk/random_pk/dgp

# Start the loop over parameter values from myfile.txt
a=1
for iteration in $(seq 1 1 9999)
do

FILE=${iteration}.txt

if [ -f "$FILE" ]
then
  n=1
##  b+="${iteration},"
  a=$((a+1))
else
   echo "$FILE does not exist"
    b+="${iteration},"
  ##    a=$((a+1))
fi
done
echo "$a"
echo "$b"

echo "DONE!!! :D "
