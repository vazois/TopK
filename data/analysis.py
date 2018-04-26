import sys
from matplotlib import pyplot

from scipy.stats import spearmanr
from itertools import combinations

def read_csv(fname,n,d):
    fp = open(fname,'r')
    cdata= [[0 for i in range(n)] for j in range(d)]
    
    print "Loading file ... "
    lines = fp.readlines()
    for i in range(len(lines)):
        ll = lines[i].strip().split(",")
        for j in range(d):
            cdata[j][i]=float(ll[j])
    
    print "Finish loading file ... "
    fp.close()
    return cdata
    
def find_correlation(cdata,n,d):
    #rho,pval=spearmanr([1,2,3,4,5],[5,6,7,8,7])
    qq=[q for q in range(2,d+1,1)]
    print qq
    #for q in qq:
    cmb = [m for m in combinations([i for i in range(d)], 2)]
    print "cmb: ",cmb
    
    acc_rho = 0
    size = len(cmb)
    for c in cmb:
        rho,pval=spearmanr(cdata[c[0]],cdata[c[1]])
        acc_rho+=rho
        print "rho: ",rho,", pval:",pval
    print "acc_rho: ", (acc_rho/len(cmb))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage:",sys.argv[0],"<file>"
        exit(1)
    
    fname = sys.argv[1]
    print "Process file: ", fname
    info = fname.strip().split("_")
    print info
    n = int(info[1])
    d = int(info[2])
    
    cdata = read_csv(fname,n,d)
    find_correlation(cdata,n,d)
