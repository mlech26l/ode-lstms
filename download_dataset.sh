#!/bin/bash

mkdir data
mkdir data/person

wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt
mv ConfLongDemo_JSI.txt data/person/

wget https://www.mit.edu/~mlechner/walker.zip
unzip walker.zip -d data/
rm walker.zip
