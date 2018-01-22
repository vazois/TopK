#!/bin/bash

#cd data/; python skydata.py $N $D $distr ; cd .. ; make cpu_cc ; ./cpu_run -f=data/$fname

START_N=$((1*1024*1024))
END_N=$((256*1024*1024))
START_D=8
END_D=8

distr=i
#bench=0#0:NA, 1:FA, 2:TA, 3:BPA, 4:CBA
#CPU:0,GPU:1
device=0

#COMPILE CPU OR GPU
if [ $device -eq 0 ]
then
  make cpu_cc
else
  make gpu_cc
fi

#CREATE DATA FILE FOR GIVEN DIMENSION
fname='d_'$END_N'_'$END_D'_'$distr
if [ ! -f data/$fname ]; then
	echo "Creating file <"$fname">"
	cd data/; time python skydata.py $END_N $END_D $distr ; cd ..
fi

#BENCH START_N TO END_N CARDINALITY
./cpu_run -f=data/$fname -n=$END_N -d=$END_D -nl=$START_N -nu=$END_N
	