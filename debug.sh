#!/bin/bash

#cd data/; python skydata.py $N $D $distr ; cd .. ; make cpu_cc ; ./cpu_run -f=data/$fname

START_N=$((16*1024*1024))
END_N=$((16*1024*1024))
START_D=4
END_D=4

distr=i
#bench=0#0:NA, 1:FA, 2:TA, 3:BPA, 4:CBA
#CPU:0,GPU:1
device=0

if [ $device -eq 0 ]
then
  make cpu_cc
else
  make gpu_cc
fi


for (( n=$START_N; n<=$END_N; n*=2 ))
do
	for (( d=$START_D; d<=$END_D; d+=2 ))
	do
		fname='d_'$n'_'$d'_'$distr
		#echo "Processing ... "$fname
		if [ ! -f data/$fname ]; then
    		echo "Creating file <"$fname">"
			cd data/; python skydata.py $n $d $distr ; cd ..
		fi
		
		#make cpu_cc ; ./cpu_run -f=data/$fname -n=$n -d=$d
		#make gpu_cc ; ./gpu_run -f=data/$fname -n=$n -d=$d
		
		if [ $device -eq 0 ]
		then
  			./cpu_run -f=data/$fname -n=$n -d=$d
		else
  			./gpu_run -f=data/$fname -n=$n -d=$d
		fi
		
		if [ $? -eq 1 ]
		then
			echo "error occured!!!"
			exit
		fi
		sleep 1
		
	done
done