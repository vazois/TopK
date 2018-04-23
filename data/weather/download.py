import sys
import urllib
from subprocess import call

import time

station = dict()

def extract_csv(fname):
    global station
    
    print "Loading file ...(",fname,")"
    with open(fname,'r') as fp:
        for line in fp:
            data = line.strip().split(",")
            
            if data[2] == "TMAX" or data[2] == "TMIN":
                sid = data[0]
                #print sid
                if sid not in station:
                    station[sid] = dict()
                
                type = data[2]
                if type not in station[sid]:
                    station[sid][type] = dict()
                
                date = data[1][4:]
                if date not in station[sid][type]:
                    station[sid][type][date] = list()
                station[sid][type][date].append(float(data[3])/10)
                    
    print "Finish loading file ... "
    return station

def fetch_files(syear,eyear):
    URL="https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/"
    OUT="out.csv"
    
    for year in range(syear,eyear+1):
        start = time.time()
        FILE=str(year)+".csv.gz"
        
        print "Downloading", FILE, "..."
        arg_call = ["wget", "-q","--show-progress",URL+FILE]
        call(arg_call)
        
        print "Unpacking", FILE, "..."
        arg_call = ["gunzip", FILE]
        call(arg_call)
        
        FILE=str(year)+".csv"
        print "Extracting temperature from",FILE,"..."
        arg_call = ["cat "+FILE+"|"+"grep"+" -e "+"TMAX"+" -e"+" TMIN"+" > "+OUT]
        call(arg_call,shell=True)
        
        print "Deleting",FILE,"..."
        arg_call = ["rm","-rf",FILE]
        call(arg_call)
        
        print "Extracting csv (",OUT,")"
        extract_csv(OUT)
        
        elapsed = time.time() - start
        print "Elapsed time: ",elapsed
        
    print "Deleting",OUT,"..."
    arg_call = ["rm","-rf",OUT]
    call(arg_call)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage:",sys.argv[0],"<start_year> <end_year>"
        exit(1)
    
    syear=int(sys.argv[1])
    eyear=int(sys.argv[2])
    
    fetch_files(syear,eyear)

# testfile = urllib.URLopener()
# testfile.retrieve("http://randomsite.com/file.gz", "file.gz")
