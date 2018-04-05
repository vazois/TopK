import time
from heapq import nlargest
from itertools import combinations

from partition_data import tuple
from partition_data import random_partitioned_data
from partition_data import bin_tree_partitioned_data
from partition_data import bin_tree_partitioned_data2
from partition_data import angle_partitioned_data
from partition_data import angle_partitioned_data2

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
                    qqs.append(tuple(t.id,score))
                elif qqs[0].score < score:
                    qqs[0] = tuple(t.id,score)
                qqs = sorted(qqs, key=lambda qq: qq.score, reverse = False)
                
                tset.add(t.id)
        
        if((qqs[0].score) >=  threshold and len(qqs) >= k):
            stop = True
            break
        
    return [qqs[0],objects_fetched,qqs]

def TA2(db,qq,k):#Threshold aggregation#
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
        #for a in qq:
        for j in range(d):
            #t = lists[a][i]
            t = lists[j][i]
            threshold+=t.score
            #if t.id not in tset:
            if t.id not in tset and (j in qq):
                objects_fetched+=1
                score = 0
                #for m in range(qq):
                #   score+=data[t.id][m]
                for a in qq:
                    score+=data[t.id][a]
                           
                if len(qqs) < k:
                    qqs.append(tuple(t.id,score))
                elif qqs[0].score < score:
                    qqs[0] = tuple(t.id,score)
                qqs = sorted(qqs, key=lambda qq: qq.score, reverse = False)
                
                tset.add(t.id)
        
        if((qqs[0].score) >=  threshold and len(qqs) >= k):
            stop = True
            break
        
    return [qqs[0],objects_fetched,qqs]


def BTA(db,q,k,part_type):
    parts=[]
    if part_type==0:
        print "Random Partitioned!!!"
        parts = random_partitioned_data(db,32)
    elif part_type==1:
        print "Bin Tree Partitioned!!!"
        parts = bin_tree_partitioned_data(db)
    elif part_type==2:
        print "Angle Tree Partitioned!!!"
        parts = angle_partitioned_data(db,2)
    elif part_type==3:
        print "Angle Tree Partitioned 2!!!"
        parts = angle_partitioned_data2(db,2)
    #parts = angle_partitioned_data3(db,4)
    
    part_count = 0
    db_parts=[]
    for part in parts:
         db_part = create_lists(part)
         db_parts.append(db_part)
         if db_part[0] > 0:
             part_count+=1
    
    for qq in q:
        cmb = [m for m in combinations([i for i in range(db[1])], qq)]
        avg_objects_fetched = 0
        min_objects_fetched = db[0]
        for c in cmb:
            #part_count = 0
            objects_fetched=0
            qqs=[]
            tt = 0

            i = 0
            for db_part in db_parts:
                if db_part[0] > 0:
                    start= time.time()
                    info=TA(db_part,c,k)
                    tt+=time.time() - start
                    #print "[ "+str(i)+" ] Fetched: ", info[1],"out of", db_part[0],"["+str(float(info[1])/db_part[0])+"]"
                    objects_fetched+=info[1]
                    qqs = qqs + info[2]
                    #part_count+=1
                i+=1
        
            qqs = sorted(qqs, key=lambda qq: qq.score,reverse=True)
            
            avg_objects_fetched+=objects_fetched
            min_objects_fetched = min(min_objects_fetched,objects_fetched)
            info=[qqs[k-1],objects_fetched,tt]
            print "<BTA>: ("+str(qq)+"D)",c,"[ threshold =",info[0],"] , [ accesses =",info[1],"] , [ tt = ", info[2] ," ]", "pcount =", part_count
        print "<BTA>: AVG ("+str(qq)+"D)", int(round(float(avg_objects_fetched)/len(cmb))),
        print "<BTA>: MIN ("+str(qq)+"D)", min_objects_fetched,"pcount =", part_count, "part_num =",len(parts)
        