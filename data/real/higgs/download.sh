#!/bin/bash

URL=http://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
FNAME=HIGGS.csv.gz

echo "Downloading data ... "
#wget -q --show-progress $URL
sleep 1

echo "Unpacking file ... "
#gunzip $FNAME
FNAME=HIGGS.csv

OUTPUT=h_22000000_5
echo "Extracting columns ... "
awk -F "\"*,\"*" '{print $2","$7","$12","$16","$22}' $FNAME > $OUTPUT
awk -F "\"*,\"*" '{print $5","$9","$14","$19","$24}' $FNAME >> $OUTPUT
#awk -F "\"*,\"*" '{print $1","$6","$11","$12","$14","$22","$25","$28}' $FNAME > $OUTPUT
#awk -F "\"*,\"*" '{print $3","$8","$15","$27","$19","$20","$24","$27}' $FNAME >> $OUTPUT
#awk -F "\"*,\"*" '{print $9","$10","$11","$12}' $FNAME >> $OUTPUT
#awk -F "\"*,\"*" '{print $13","$14","$15","$16}' $FNAME >> $OUTPUT
#awk -F "\"*,\"*" '{print $17","$18","$19","$20}' $FNAME >> $OUTPUT
#awk -F "\"*,\"*" '{print $21","$22","$23","$24}' $FNAME >> $OUTPUT
#awk -F "\"*,\"*" '{print $25","$26","$27","$28}' $FNAME >> $OUTPUT

#rm -rf $FNAME




