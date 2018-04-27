#!/bin/bash

URL=https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_
START_YEAR=$1
MONTH_COUNT=$2
DATE=$START_YEAR-01-01

if (( $START_YEAR < 2009 ))
then
	echo "Data only after 2009!!!"
	exit 1
fi

for (( n=0; n<$MONTH_COUNT; n+=1 ))
do
	NEXT_DATE=$(date +%Y-%m -d "$DATE + $n month")
	#echo "$NEXT_DATE"
	
	FILE=$NEXT_DATE".csv"
	echo "Downloading "$FILE" ... "
	#wget -q --show-progress $URL$FILE

done