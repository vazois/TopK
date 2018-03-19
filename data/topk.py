import sys
import math
from itertools import combinations
from itertools import permutations

from algorithms import create_lists
from algorithms import TA
from algorithms import FA
from algorithms import BTA

from partition_data import read_file 
from partition_data import slope_tree_partitioned_data

import time

import math

if __name__ == "__main__":
    if len(sys.argv) <2:
        print "Execute:",sys.argv[0],"<file>"
        exit(1)

    k=100
    db=read_file(sys.argv[1])

    qq=[q for q in range(2,db[1]+2,2) ]
    print "<<<",db[0],qq,">>>"
    for q in qq:
        info=FA(db,q,k)
        print "<FA>: ("+str(q)+"D) [ threshold =",info[0],"] , [ accesses =",info[1],"]"
    
    db0=create_lists(db)
#     for q in qq:
#         cmb = [m for m in combinations([i for i in range(db[1])], q)]
#         print "("+str(q)+"D)"
#         for c in cmb:
#             tt = time.time()
#             info=TA(db0,c,k)
#             tt = time.time() - tt 
#             print "<TA>: [ threshold =",info[0],"] , [ accesses =",info[1],"] , [ tt = ", tt ," ]"
    
    info=BTA(db,qq,k)

    