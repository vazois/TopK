import sys
import math
from itertools import combinations
from itertools import permutations

from algorithms import create_lists
from algorithms import TA
from algorithms import FA
from algorithms import BTA

from partition_data import read_file
import time
import math

from partition_data import angle_partitioned_data2

if __name__ == "__main__":
    if len(sys.argv) <2:
        print "Execute:",sys.argv[0],"<file>"
        exit(1)

    k=100
    db=read_file(sys.argv[1])
#     parts = angle_partitioned_data2(db,4)
#     exit(1)

    #qq=[q for q in range(2,db[1]+1,1) ]
    qq=[db[1]]
    print "<<<",db[0],qq,">>>"
#     for q in qq:
#         info=FA(db,q,k)
#         print "<FA>: ("+str(q)+"D) [ threshold =",info[0].id,",{0:.4f}".format(info[0].score),"] , [ accesses =",info[1],"]"
     
#     db0=create_lists(db)
#     for q in qq:
#         cmb = [m for m in combinations([i for i in range(db[1])], q)]
#         avg_objects_fetched=0
#         for c in cmb:
#             tt = time.time()
#             info=TA(db0,c,k)
#             tt = time.time() - tt 
#             avg_objects_fetched+=info[1]
#             print "<TA>: ("+str(q)+"D)",c,"[ threshold =",info[0].id,",{0:.4f}".format(info[0].score),"] , [ accesses =",info[1],"] , [ tt = ", tt ," ]"
#         print "<TA>: AVG("+str(q)+"D)",int(round(float(avg_objects_fetched)/len(cmb)))
#     db0=[]
    
#     info=BTA(db,qq,k,0)
    #info=BTA(db,qq,k,1)
#     info=BTA(db,qq,k,2)
    info=BTA(db,qq,k,3)

    