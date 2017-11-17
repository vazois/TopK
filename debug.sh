#!/bin/bash

N=$((1024 * 1024))
D=6
distr=i
fname='d_'$N'_'$D'_'$distr

echo "Processing ... "$fname

cd data/; python skydata.py $N $D $distr ; cd .. ; make cpu_cc ; ./cpu_run -f=data/$fname