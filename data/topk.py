import sys
import math
from itertools import combinations
from itertools import permutations

from algorithms import create_lists
from algorithms import TA
from algorithms import FA
from algorithms import BTA

from partition_data import read_file
from partition_data import spherical_partitioned_data      
import time

import math


if __name__ == "__main__":
    if len(sys.argv) <2:
        print "Execute:",sys.argv[0],"<file>"
        exit(1)

    k=100
    #qq=2
    db=read_file(sys.argv[1])
    for qq in range(2,6,2):
        print "<<<",db[0],qq,">>>"
        #print "Find TopK FA!!!"
        info=FA(db,qq,k)
        print "<FA>: [ threshold =",info[0],"] , [ accesses =",info[1],"]"
    
        #print "Find TopK TA!!!"
        db0=create_lists(db)
        tt = time.time()
        info=TA(db0,qq,k)
        tt = time.time() - tt 
        print "<TA>: [ threshold =",info[0],"] , [ accesses =",info[1],"] , [ tt = ", tt ," ]"
    
        #print "Find TopK BTA!!!"
        info=BTA(db,qq,k)
        print "<BTA>: [ threshold =",info[0],"] , [ accesses =",info[1],"] , [ tt = ", info[2] ," ]"

    