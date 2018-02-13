import sys
import numpy as np

def process_const_file(filename,n,d):
    lists=[[0 for j in range(n)] for i in range(d)]
    print "Loading Data !!!"
    with open(filename) as fp:
        i = 0
        for line in fp:
            data = line.strip().split(",")
            for j in range(d):
                lists[j][i]=float(data[j])
            i+=1
    print "Finished Loading !!!"
    return lists


def histogram(list,bins):
    pair = np.histogram(list,bins,density=False)
    #print pair
    return pair[0]
    
def gather_stats(lists):
    bins=[]
    v = 0.0
    while(v <= 1.0):
        bins.append(v)
        v+=0.1
    print "bins:[",",".join([str(v) for v in bins]),"]"
    
    d = len(lists)
    for i in range(d):
        h = histogram(lists[i],bins)
        print str(i),":[",",".join([str(v) for v in h]),"]"

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Usage: " + sys.argv[0] + " <filename> [rows] [attributes] ")
        exit()
    filename = sys.argv[1]
    n = 0
    d = 0
    if(len(sys.argv) >= 2):
        n =int(sys.argv[2])
        d =int(sys.argv[3])

    lists=process_const_file(filename,n,d)
    gather_stats(lists)
    
    
    
