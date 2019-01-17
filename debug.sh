#!/bin/bash

#############################
###### DATA PARAMETERS ######
#############################
START_N=$((256*1024*1024))
END_N=$((256*1024*1024))
DIMS=8
#Top-K Range in power of 2 (i.e. KKS = 16 , KKS = 128 .. k=16,32,64,128)
KKS=16
KKE=16
#LD 0:load from file, 1: generate in memory, 2: Load real data (set REAL_DATA_PATH)
LD=1

#distr c:correlated i:independent a:anticorrelated
distr=i
#randdataset3: higher precission , randdataset: lower precission
script=randdataset

######################
### DO NOT CHANGE  ###
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
######################

#REAL DATA PARAMETERS
REAL_DATA_PATH=data/real/weather/wtuples_261552285_8
#REAL_DATA_PATH=data/real/higgs/h_11000000_5
#REAL_DATA_N=$START_N
REAL_DATA_N=$(echo $REAL_DATA_PATH| cut -d'_' -f 2)
#REAL_DATA_N=256000000
REAL_DATA_D=$(echo $REAL_DATA_PATH| cut -d'_' -f 3)
#echo "REAL_DATA: "$REAL_DATA_N","$REAL_DATA_D
#exit 1
if [ $LD -eq 2 ]
then
	DIMS=$REAL_DATA_D
fi

###################################
###### EXPERIMENT PARAMETERS ######
###################################
#CPU:0,GPU:1
device=1
#QM 0:Reverse query attribute, 1:Forward query attributes
QM=0
#QD Dimension interval for testing
QD=1
#IMP 0:Scalar, 1:SIMD, 2:Threads, 3:Multiple Queries (Random), 4: Multiple Queries (Same dimension)
IMP=1
#ITER Testing iterations
ITER=10
#Multiple Thread Count
MQTHREADS=16
#Gather object evaluation statistics
STATS_EFF=false
#Choose workload for multi-query evaluation
WORKLOAD=$((1024*128))

if [ ! -z $1 ]
then
    MQTHREADS=$1
fi

######CHOOSE CPU ALGORITHM######
#TA Benchmark
TA_B=0
#TPAc Benchmark
TPAc_B=0
#TPAr Benchmark
TPAr_B=0
#VTA Benhmark
VTA_B=0
#PTA Benchmark
PTA_B=1
#SLA Benchmark
SLA_B=0
#####################################################################################	

####################################
###### COMPILATION PARAMETERS ######
####################################
if [ $device -eq 0 ]
then
	make cpu_cc DIMS=$DIMS QM=$QM QD=$QD IMP=$IMP ITER=$ITER LD=$LD DISTR=$DSTR TA_B=$TA_B TPAc_B=$TPAc_B TPAr_B=$TPAr_B VTA_B=$VTA_B PTA_B=$PTA_B SLA_B=$SLA_B KKS=$KKS KKE=$KKE MQTHREADS=$MQTHREADS STATS_EFF=$STATS_EFF WORKLOAD=$WORKLOAD
else
	make gpu_cc DIMS=$DIMS QM=$QM QD=$QD IMP=$IMP ITER=$ITER LD=$LD DISTR=$DSTR KKS=$KKS KKE=$KKE STATS_EFF=$STATS_EFF WORKLOAD=$WORKLOAD
fi
#################################################################################


#############################################
###### EXPERIMENT EXECUTION PARAMETERS ######
#############################################
if [ $LD -eq 2 ]
then
	echo "<--------------USING REAL DATA PATH----------------->"
	if [ $device -eq 0 ]
	then
		echo "./cpu_run -f=$REAL_DATA_PATH -n=$REAL_DATA_N -d=$DIMS"
  		./cpu_run -f=$REAL_DATA_PATH -n=$REAL_DATA_N -d=$DIMS
	else
  		./gpu_run -f=$REAL_DATA_PATH -n=$REAL_DATA_N -d=$DIMS
	fi
	echo "<--------------------------------------------------->"
	if [ $? -eq 1 ]
	then
		echo "error occured!!!"
		exit
	fi
else
	for (( n=$START_N; n<=$END_N; n*=2 ))
	do
		fname='d_'$n'_'$DIMS'_'$distr
		#echo "Processing ... "$fname
		if [ ! -f data/$fname ] && [ $LD -eq 0 ] 
		then
    		echo "Creating file <"$fname">"
			cd data/; python skydata.py $n $DIMS $distr $script; cd ..
			#cd data/; time python rand.py $n $d $distr; cd ..
		fi
		
		if [ $device -eq 0 ]
		then
			echo "./cpu_run -f=data/$fname -n=$n -d=$DIMS"
  			./cpu_run -f=data/$fname -n=$n -d=$DIMS
		else
  			#nvprof --devices 0 ./gpu_run -f=data/$fname -n=$n -d=$DIMS
			#echo "./gpu_run -f=data/$fname -n=$n -d=$DIMS"
			set -e
			for (( x=0; x<1; x+=1 ))
			do
				./gpu_run -f=data/$fname -n=$n -d=$DIMS
			done
		fi
		
		if [ $? -eq 1 ]
		then
			echo "error occured!!!"
			exit
		fi
		sleep 1
		#rm -rf data/$fname
		#echo "<--------------------------------------------------->"
	done
fi