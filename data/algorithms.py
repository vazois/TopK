import time
from heapq import nlargest
from itertools import combinations

from partition_data import tuple
from partition_data import bin_tree
from partition_data import bin_tree_partitioned_data
from partition_data import bin_tree_partitioned_data2
from partition_data import slope_tree_partitioned_data
from partition_data import spherical_partitioned_data

def create_lists(db):
    n = db[0]
    d = db[1]
    data = db[2]
    
    #print "create lists: <", n, d,">"
    lists = []
    for m in range(d):
        alist = []
        for i in range(n):
            alist.append(tuple(i,data[i][m]))
        alist = sorted(alist, key=lambda tuple: tuple.score,reverse=True)
        #print alist[0]
        lists.append(alist)
        
    return [db[0],db[1],db[2],lists]

def FA(db,qq,k):#Full Aggregation
    n = db[0]
    d = db[1]
    data = db[2]
    
    scores = []
    for i in range(n):
        score = 0
        for m in range(qq):
            score+=data[i][m]
        scores.append(score)
    nl = nlargest(k,scores)
    #print "Threshold: ",min(nl)
    return [min(nl),n,nl]

def TA(db,qq,k):#Threshold aggregation#
    n = db[0]
    d = db[1]
    data = db[2]
    lists = db[3]

    
    objects_fetched = 0
    stop = False
    tset = set()
    qqs = []
    for i in range(n):
        threshold = 0
        for a in qq:
            t = lists[a][i]
            threshold+=t.score
            if t.id not in tset:
                objects_fetched+=1
                score = 0
                #for m in range(qq):
                #   score+=data[t.id][m]
                for a in qq:
                    score+=data[t.id][a]
                           
                if len(qqs) < k:
                    qqs.append((t.id,score))
                elif qqs[0][1] < score:
                    qqs[0] = (t.id,score)
                qqs = sorted(qqs, key=lambda qq: qq[1])
                tset.add(t.id)
                
        
        if((qqs[0][1]) >=  threshold and len(qqs) >= k):
            stop = True
            break
    
#     print "Threshold: ",qqs[0][1]
#     print "Objects fetched: ", objects_fetched
    return [qqs[0],objects_fetched,qqs]

def BTA(db,q,k):
    parts = bin_tree_partitioned_data2(db)
    #parts = spherical_partitioned_data(db,16)
    #parts = slope_tree_partitioned_data(db)
    
    db_parts=[]
    for part in parts:
         db_part = create_lists(part)
         db_parts.append(db_part)
         
    for qq in q:
        cmb = [m for m in combinations([i for i in range(db[1])], qq)]
        print "("+str(qq)+"D)"
        for c in cmb:
            objects_fetched=0
            qqs=[]
            tt = 0

            i = 0
            for db_part in db_parts:
                if db_part[0] > 0:
                    start= time.time()
                    info=TA(db_part,c,k)
                    tt+=time.time() - start
                    print "[ "+str(i)+" ] Fetched: ", info[1]
                    objects_fetched+=info[1]
                    qqs = qqs + info[2]
                i+=1
        
            qqs = sorted(qqs, key=lambda qq: qq[1],reverse=True)
            info=[qqs[k],objects_fetched,tt]
            print "<BTA>: [ threshold =",info[0],"] , [ accesses =",info[1],"] , [ tt = ", info[2] ," ]"
    