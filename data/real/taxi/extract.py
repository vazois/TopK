import sys
import os

OUTPUT="taxi_data.csv"

def normalize(v,MMAX,MMIN):
    return (v - MMIN)/(MMAX-MMIN)

def build_csv(fname):
    global OUTPUT
    outp = open(OUTPUT,"w")
    with open(fname,'r') as fp:
        line=fp.readline()#SKIP FIRST TWO LINES
        fp.readline()
        items = len(line.strip().split(" "))
        rows = 0
        for line in fp:
            data = line.strip().split(" ")
            data = ",".join([str(float(v)) for v in data])
            outp.write(data+"\n")    
            rows+=1
            
    outp.close()
    print rows,items
    os.rename(OUTPUT,"taxi_"+str(rows)+"_"+str(items))
    
def build_norm_csv(fname,d):
    global OUTPUT
    cdata=list()
    mmax = []
    mmin = []
    print "Loading file ..."
    with open(fname,'r') as fp:
        line=fp.readline()#SKIP FIRST TWO LINES
        fp.readline()
        items = len(line.strip().split(" "))
        rows = 0
        cdata = [[] for i in range(items)]
        mmax = [0 for i in range(items)]
        mmin = [1024*1024*1024 for i in range(items)]
        
        for line in fp:
            data = line.strip().split(" ")
            for j in range(items):
                v = float(data[j])
                mmax[j] = max(mmax[j],v)
                mmin[j] = max(mmin[j],v)
                cdata[j].append(v)
            rows+=1
    
    print "Normalizing values ... "        
    outp = open(OUTPUT,"w")
    print len(cdata),len(cdata[0])
    for i in range(rows):
        data = []
        for j in range(items):
            data.append( str(normalize(cdata[j][i],mmax[j],mmin[j])) )
        outp.write( ",".join(data) +  "\n")
    outp.close()
    os.rename(OUTPUT,"taxi_"+str(rows)+"_"+str(items))
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage:",sys.argv[0],"<file>"
        exit(1)
        
    fname = sys.argv[1]
    print "Input file:",fname
    build_csv(fname)
    #build_norm_csv(fname)
    
    