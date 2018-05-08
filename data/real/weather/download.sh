#!/bin/bash

URL=https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/
START_YEAR=$1
END_YEAR=$2
D=$3
LIMIT=$((END_YEAR-START_YEAR+1))

OUTPUT=$START_YEAR"_"$END_YEAR".csv"

rm -rf $OUTPUT
TMP="tmp.csv"

for (( n=$START_YEAR; n<=$END_YEAR; n+=1 ))
do
	FILE=$n".csv.gz"
	echo "Downloading "$FILE" ... "
	wget -q --show-progress $URL$FILE
	
	echo "Unpacking "$FILE" ... "
	gunzip $FILE
	rm -rf $FILE
	
	echo "Merging with "$OUTPUT" ... "
	FILE=$n".csv"
	awk -F "\"*,\"*" '$3=="TMAX" || $3=="TMIN"{print $1","$3","$4}' $FILE >> $OUTPUT
	#awk -F "\"*,\"*" '$3=="TMAX" || $3=="TMIN"{print $1","$2","$3","$4}' $FILE >> $OUTPUT
	#awk -F "\"*,\"*" '$3=="TMAX" || $3=="TMIN" || $3=="WSF2" || $3=="WSF5" {print $1","$2","$3","$4}' $FILE >> $OUTPUT
	#awk -F "\"*,\"*" '$3=="WSF2" || $3=="WSF5"{print $1","$2","$4}' $FILE >> $OUTPUT
	rm -rf $FILE
done

n=($(wc -l $OUTPUT))
echo $n
mv $OUTPUT "w_"$n"_1"