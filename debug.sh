#!/bin/bash

#cd data/; python skydata.py $N $D $distr ; cd .. ; make cpu_cc ; ./cpu_run -f=data/$fname

START_N=$((16*1024*1024))
END_N=$((16*1024*1024))
START_D=2
END_D=16

distr=p
#bench=0#0:NA, 1:FA, 2:TA, 3:BPA, 4:CBA
#CPU:0,GPU:1
device=$1
#randdataset3: higher precission , randdataset: lower precission
script=randdataset
#mem:1 only for memory benchmark, do not create file
mem=0

if [ $device -eq 0 ]
then
  make cpu_cc
else
  make gpu_cc
fi

make reorder_cc


for (( n=$START_N; n<=$END_N; n*=2 ))
do
	for (( d=$START_D; d<=$END_D; d+=2 ))
	do
		fname='d_'$n'_'$d'_'$distr
		fname2='d_'$n'_'$d'_'$distr'_o'
		#echo "Processing ... "$fname
		if [ ! -f data/$fname ] && [ $mem -eq 0 ] 
		then
    		echo "Creating file <"$fname">"
			#cd data/; python skydata.py $n $d $distr $script; cd ..
			cd data/; time python rand.py $n $d $distr; cd ..
		fi
		if [ ! -f data/$fname2 ]
		then
			echo "Reorder data <"$fname2">"
			#./reorder_run -f=data/$fname -n=$n -d=$d
		fi
		
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
		#rm -rf data/$fname
		echo "<--------------------------------------------------->"
	done
done