#!/bin/bash

#cd data/; python skydata.py $N $D $distr ; cd .. ; make cpu_cc ; ./cpu_run -f=data/$fname

START_N=$((16*1024 * 1024))
END_N=$((16*1024 * 1024))
START_D=8
END_D=8

distr=i

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
		
		#make cpu_cc ; ./cpu_run -f=data/$fname
		make gpu_cc ; ./gpu_run -f=data/$fname
		
		if [ $? -eq 1 ]
		then
			echo "error occured!!!"
			exit
		fi
		sleep 1
		
	done
	
done