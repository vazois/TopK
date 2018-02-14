#!/bin/bash

START_N=$((1*1024*1024))
END_N=$((1*1024*1024))
START_D=2
END_D=2

distr=p

for (( n=$START_N; n<=$END_N; n*=2 ))
do
	for (( d=$START_D; d<=$END_D; d+=2 ))
	do
		fname='d_'$n'_'$d'_'$distr
		if [ ! -f $fname ] 
		then
			echo "Creating file <"$fname">"
			time python rand.py $n $d $distr
		fi
		
		#sleep 1
		#time python mstats.py $fname $n $d
		
		rm -rf $fname
	done
done
