import time
from heapq import nlargest

from partition_data import tuple
from partition_data import bin_tree
from partition_data import bin_tree_partitioned_data
from partition_data import bin_tree_partitioned_data2
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
        for j in range(qq):
            t = lists[j][i]
            threshold+=t.score
            if t.id not in tset:
                objects_fetched+=1
                score = 0
                for m in range(qq):
                    score+=data[t.id][m]
                           
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

def BTA(db,qq,k):
    parts = bin_tree_partitioned_data2(db)
    #parts = spherical_partitioned_data(db,16)
    
    objects_fetched=0
    qqs=[]
    tt = 0

    for part in parts:
        if part[0] > 0:
            db_part = create_lists(part)
            start= time.time()
            info=TA(db_part,qq,k)
            tt+=time.time() - start
            
            #print "Fetched: ", info[1]
            objects_fetched+=info[1]
            qqs = qqs + info[2]
    
    qqs = sorted(qqs, key=lambda qq: qq[1],reverse=True)
    
    return [qqs[k],objects_fetched,tt]