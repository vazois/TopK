import sys
import random
import numpy as np
import time
import struct

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
        #fp.write(ret_distr(bsize,d,distr,sample))
        fp.write(scale_sample(bsize,d,sample))
        i = i + bsize  
    tt = time.time() - tt
    
    print "tt: ",tt
    
    fp.close()

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print("Usage: " + sys.argv[0] + " <cardinality> <attributes> <distribution: (u)niform>")
        exit()
    n = int(sys.argv[1])
    d = int(sys.argv[2])
    distr = "z"
    if (len(sys.argv) == 4):
        distr=sys.argv[3]
        
    ZIPF_ALPHA=2
    _MAX=2*1024*1024
    p = 0.1
    FORMAT="{:0.4f}"
        
    create_file(n,d,distr)

#print next_value(distr)

# tt=time.time()
# 
# fname= "d_"+str(n)+"_"+str(d)+"_"+distr
# fp = open(fname,'w')
# arr=[decrease(distr,d)[:-1]+"\n" for i in range(n)]
# fp.write("".join(arr))
# fp.close()
# 
# 
# tt = time.time() - tt

# print "tt:",tt

# fname= "d_"+str(n)+"_"+str(d)+"_"+distr
# fp = open(fname,'w')
# 
#     
# rr=1024*1024
# rows = 0
# while(rows < n):
#     lines=""
#     gen_rows = rr
#     if rows + rr >= n:
#         gen_rows = n - rows
#     
#     for i in range(rr):
#         line=decrease(distr,d)
#         lines+=line[:-1]+"\n"
#     fp.write(lines)
#     rows = rows + rr
#     
# fp.close()