#!/bin/bash
echo "Starting ...."
echo ""

cd "$(dirname "$0")"

# Specify directory with files to be checked
cd /Users/bbose/Desktop/ML+ReACT/reactions/examples/randpk/random_pk/dgp

# Run check over indices 1 to 9999 
a=1
for iteration in $(seq 1 1 9999)
do

FILE=${iteration}.txt

if [ -f "$FILE" ]
then
  n=1
  a=$((a+1))
else
   echo "$FILE does not exist"
    b+="${iteration},"
fi
done
echo "$a"
echo "$b"

echo "DONE!"
