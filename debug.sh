#!/bin/bash

#cd data/; python skydata.py $N $D $distr ; cd .. ; make cpu_cc ; ./cpu_run -f=data/$fname

START_N=$((1*1024*1024))
END_N=$((1*1024*1024))
DIMS=8

#QM 0:Multiple dimension queries, 1:Single dimension Queries
QM=0
#QD Dimension interval for testing
QD=1
#IMP 0:Scalar, 1:SIMD, 2:Threads
IMP=1
#ITER Testing iterations
ITER=1
#LD 0:load from file, 1: generate in memory
LD=0

#distr c:correlated i:independent a:anticorrelated
distr=a
#bench=0#0:NA, 1:FA, 2:TA, 3:BPA, 4:CBA
#CPU:0,GPU:1
device=$1
#randdataset3: higher precission , randdataset: lower precission
script=randdataset

DSTR=0
if [ $distr == "c" ]
then
	DSTR=0
elif [ $distr == "i" ]
then
	DSTR=1
elif [ $distr == "a" ]
then
	DSTR=2
else
	echo "DSTR not valid option!!!!"
	exit
fi

if [ $device -eq 0 ]
then
	make cpu_cc DIMS=$DIMS QM=$QM QD=$QD IMP=$IMP ITER=$ITER LD=$LD DISTR=$DSTR
else
	make gpu_cc
fi



for (( n=$END_N; n<=$END_N; n*=2 ))
do
		fname='d_'$n'_'$DIMS'_'$distr
		fname2='d_'$n'_'$DIMS'_'$distr'_o'
		#echo "Processing ... "$fname
		if [ ! -f data/$fname ] && [ $LD -eq 0 ] 
		then
    		echo "Creating file <"$fname">"
			cd data/; python skydata.py $n $DIMS $distr $script; cd ..
			#cd data/; time python rand.py $n $d $distr; cd ..
		fi
		
		if [ $device -eq 0 ]
		then
  			./cpu_run -f=data/$fname -n=$n -d=$DIMS
		else
  			./gpu_run -f=data/$fname -n=$n -d=$DIMS
		fi
		
		if [ $? -eq 1 ]
		then
			echo "error occured!!!"
			exit
		fi
		sleep 1
		#rm -rf data/$fname
		echo "<--------------------------------------------------->"
done