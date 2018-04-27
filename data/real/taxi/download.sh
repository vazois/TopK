#!/bin/bash

URL=https://s3.amazonaws.com/nyc-tlc/trip+data/
PREFIX_FNAME=yellow_tripdata_
START_YEAR=$1
MONTH_COUNT=$2
ACTION=$3

DATE=$START_YEAR-01-01
OUTPUT=all.csv
rm -rf

if (( $START_YEAR < 2009 ))
then
	echo "Data only after 2009!!!"
	exit 1
fi

for (( n=0; n<$MONTH_COUNT; n+=1 ))
do
	NEXT_DATE=$(date +%Y-%m -d "$DATE + $n month")
	#echo "$NEXT_DATE"
	
	FILE=$PREFIX_FNAME$NEXT_DATE".csv"
	
	if [ $ACTION -eq 0 ]
	then
		echo "Downloading "$FILE" ... "
		wget -q --show-progress $URL$FILE
	fi
	
	echo "Extracting columns ... "
	awk -F "\"*,\"*" '{print $5","$6}' $FILE > $OUTPUT

done
