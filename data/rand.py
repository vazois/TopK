import sys
import random
import numpy as np
import time
import struct
import numpy as np

import math
from random import gauss
from random import lognormvariate
from random import paretovariate

def FtX(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def scale(d,sample):
    global FORMAT
    line=""
    for j in range(d):
        line+=FORMAT.format(random.uniform(0.0,1.0)*sample[j])+","
    return line

def scale_sample(bsize,d,sample):
    lines=""
    
    for i in range(bsize):
        lines+=scale(d,sample)[:-1]+"\n"
    return lines
     
def create_file(n,d,distr):
    fname= "d_"+str(n)+"_"+str(d)+"_"+distr
    fp = open(fname,'w')
    
    bsize=1024
    tt=time.time()
    #buffer=[[FtX(0.0) for j in range(d)] for i in range(bsize)]
    buffer=[[0.0 for j in range(d)] for i in range(bsize)]
    
    sample=[]
    if distr == "i":
        sample=np.random.normal(0.5,0.1,d)
    elif distr == "z":
        sample=float(1.0)/np.random.zipf(1.5,d)
        
    print sample
        
    i = 0
    while( i  < n ):
        if distr == "i":
            sample=np.random.normal(0.5,0.1,d)
        elif distr == "z":
            sample=float(1.0)/np.random.zipf(1.5,d)
        #fp.write(ret_distr(bsize,d,distr,sample))
        fp.write(scale_sample(bsize,d,sample))
        i = i + bsize  
    tt = time.time() - tt
    
    print "tt: ",tt
    
    fp.close()

def create_zipf_sample(sample,d,p):
    global FORMAT
    line=""
    n = len(sample)
    pp =random.uniform(0.0,1.0)
    if pp <= p:
        for j in range(d):
            line+=FORMAT.format(sample[random.randrange(0,10)])+","
    else:
        for j in range(d):
            line+=FORMAT.format(sample[random.randrange(10,n)])+","
    return line[:-1]+"\n"

def create_log_sample(sample,d,p):
    global FORMAT
    line=""
    n = len(sample)
    pp =random.uniform(0.0,1.0)
#     if pp <= p:
#         for j in range(d):
#             line+=FORMAT.format(sample[random.randrange(0,10)])+","
#     else:
#         for j in range(d):
#             line+=FORMAT.format(sample[random.randrange(10,n)])+","
#     return line[:-1]+"\n"
    for j in range(d):
        line+=FORMAT.format(sample[random.randrange(0,n)])+","
    return line[:-1]+"\n"

def create_pareto_sample(sample,d,p):
    global FORMAT
    line=""
    n = len(sample)
    pp =random.uniform(0.0,1.0)
#     if pp <= p:
#         for j in range(d):
#             line+=FORMAT.format(sample[random.randrange(0,10)])+","
#     else:
#         for j in range(d):
#             line+=FORMAT.format(sample[random.randrange(10,n)])+","
    for j in range(d):
        line+=FORMAT.format(sample[random.randrange(0,n)])+","
    return line[:-1]+"\n"

def create_file2(n,d,distr):
    global stats_only
    
    sample=[]
    if distr == "g":
        sample= [lognormvariate(0, 1) for i in range(n)]
        m = max(sample)
        w = min(sample)
        print "m:",str(m),"w:",str(w)
        sample = [ (v - w)/(m-w) for v in sample ]
    elif distr == "z":
        sample=np.random.zipf(1.8,n)
        mx = float(max(sample))
        sample = [v / mx for v in sample]
        sample = sorted(sample,reverse=True)
    elif distr == "p":
        sample= [paretovariate(10) for i in range(n)]
        m = max(sample)
        w = min(sample)
        print "m:",str(m),"w:",str(w)
        sample = [ (v - w)/(m-w) for v in sample ]
        
    #print sorted(sample,reverse=True
    #print sample
    gather_stats(sample)
    if stats_only:
        print "No file written!!!"
        return
    else:
        if distr == "g":
            print "Creating file < logvariate > !!!"
        elif distr == "z":
            print "Creating file < zipf > !!!"
        elif distr == "p":
            print "Creating file < pareto > !!!"
    
    fname= "d_"+str(n)+"_"+str(d)+"_"+distr
    fp = open(fname,'w')
    i = 0
    if distr == "z":
        while( i  < n ):
            fp.write(create_zipf_sample(sample,d,0.1))
            i+=1
    elif distr=="g":
        while( i  < n ):
            fp.write(create_log_sample(sample,d,0.1))
            i+=1
    elif distr=="p":
        while( i  < n ):
            fp.write(create_pareto_sample(sample,d,0.1))
            i+=1

    fp.close()

def histogram(list,bins):
    pair = np.histogram(list,bins)
    #print pair
    return pair[0]

def gather_stats(lists):
    bins=[]
    v = 0.0
    while(v <= 1.0):
        bins.append(v)
        v+=0.01
    print "bins:[",",".join([str(v) for v in bins]),"]"
    
    d = len(lists)
    h = histogram(lists,bins)
    print "[",",".join([str(v) for v in h]),"]"
    #print histogram(lists,bins)    
    
if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print("Usage: " + sys.argv[0] + " <cardinality> <attributes> <distribution: (u)niform>")
        exit()
    n = int(sys.argv[1])
    d = int(sys.argv[2])
    distr = "z"
    stats_only = False
    if (len(sys.argv) == 4):
        distr=sys.argv[3]
    
    FORMAT="{:0.8f}"    
    create_file2(n,d,distr)
    
    
