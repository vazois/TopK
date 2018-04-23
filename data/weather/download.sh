#!/bin/bash

URL=https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/
START_YEAR=1960
END_YEAR=2017
D=2
LIMIT=$((END_YEAR-START_YEAR+1))

if (( LIMIT < D ))
then
	echo "Attributes should be at most END_YEAR-START_YEAR+1 ( "$LIMIT" )"
	exit
fi

OUTPUT=$START_YEAR"_"$END_YEAR".csv"
rm -rf $OUTPUT

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
	cat $FILE | grep -e "TMAX" -e "TMIN" >> $OUTPUT
	rm -rf $FILE
done

#echo "Gathering data ... "
#python extract.py $OUTPUT $D
