import sys
from Queue import PriorityQueue

from heapq import nlargest
import time

class tuple:
    def __init__(self, id, score):
        self.id = id
        self.score = score
    def __repr__(self):
        return repr((self.id, self.score))

def read_file(fname):
    info = fname.split("_")
    print info
    n = int(info[1])
    d = int(info[2])
    print "TopK -",fname,str(n),str(d)
    
    data = [[0 for i in range(d)] for j in range(n)]
    print "Loading Tuples!!!"
    fp = open(fname,'r')
    lines = fp.readlines()
    i = 0
    for line in lines:
        ll = line.strip().split(",")
        for j in range(d):
            data[i][j]=float(ll[j])
        i = i + 1
    fp.close()
    
    return [n,d,data]

def create_lists(db):
    n = db[0]
    d = db[1]
    data = db[2]
    
    print "create lists: <", n, d,">"
    lists = []
    for m in range(d):
        alist = []
        for i in range(n):
            alist.append(tuple(i,data[i][m]))
        alist = sorted(alist, key=lambda tuple: tuple.score,reverse=True)
        #print alist[0]
        lists.append(alist)
        
    return [db[0],db[1],db[2],lists]

def partitioned_data_2D(db):        
    n = db[0]
    d = db[1]
    data = db[2]
    
    data0 = []
    data1 = []
    for i in range(n):
        if data[i][0] >= data[i][1]:
            data0.append(data[i])
        else:
            data1.append(data[i])
    
    print "0:",len(data0)
    print "1:",len(data1)
    return [[len(data0),d,data0],[len(data1),d,data1]]

def partitioned_data(db):
    n = db[0]
    d = db[1]
    data = db[2]
    
    part_num = 2**(d-1)
    data_parts=[]
    for p in range(part_num):
        data_parts.append(list())
  
    if d==2:
        for i in range(n):
            if data[i][1] >= data[i][0]:
                data_parts[0].append(data[i])
            else:
                data_parts[1].append(data[i])
    elif d == 3:
        for i in range(n):
            if data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[0].append(data[i])
            elif data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[1].append(data[i])
            elif data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[2].append(data[i])
            else:
                data_parts[3].append(data[i])          
    elif d == 4:
        for i in range(n):
            if data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[0].append(data[i])
            elif data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[1].append(data[i])
            elif data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[2].append(data[i])
            elif data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[3].append(data[i])
            elif data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[4].append(data[i])
            elif data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[5].append(data[i])
            elif data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[6].append(data[i])
            else:
                data_parts[7].append(data[i])

    return [[len(data_parts[p]),d,data_parts[p]] for p in range(part_num) ]

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
    q = PriorityQueue()
    
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

def PTA_2D(parts,qq,k):#Partitioned Aggregation#
#     nl0=FA(parts[0],qq,k)
#     nl1=FA(parts[1],qq,k)
#     nl=nlargest(k,nl0[2]+nl1[2])
#     print "0:<FA-PTA>: [ threshold =",nl0[0],"] , [ accesses =",nl0[1],"]"
#     print "1:<FA-PTA>: [ threshold =",nl1[0],"] , [ accesses =",nl1[1],"]"
    #return [min(nl),0]

    db0 = create_lists(parts[0])
    db1 = create_lists(parts[1])
    tt = time.time()
    info0=TA(db0,qq,k)
    info1=TA(db1,qq,k)
    tt = time.time() - tt
#     print "0:<TA-PTA>: [ threshold =",info0[0],"] , [ accesses =",info0[1],"]"
#     print "1:<TA-PTA>: [ threshold =",info1[0],"] , [ accesses =",info1[1],"]"
    
    qqs = info0[2]+info1[2]
    qqs = sorted(qqs, key=lambda qq: qq[1],reverse=False)
    return [qqs[k],info0[1]+info1[1],tt]

def PTA(db,qq,k):
    parts = partitioned_data(db)
    
    objects_fetched=0
    qqs=[]
    tt = 0
    #print len(parts[0]),len(parts[1])
    #return
    for part in parts:
        if part[0] > 0:
            db_part = create_lists(part)
        
            start= time.time()
            info=TA(db_part,qq,k)
            tt+=time.time() - start
        
            objects_fetched+=info[1]
            qqs = qqs + info[2]
    
    qqs = sorted(qqs, key=lambda qq: qq[1],reverse=True)
    
    return [qqs[k],objects_fetched,tt]

if __name__ == "__main__":
    if len(sys.argv) <2:
        print "Execute:",sys.argv[0],"<file>"
        exit(1)
    
    k=100
    db=read_file(sys.argv[1])
    
    print "Find TopK FA!!!"
    info=FA(db,2,k)
    print "<FA>: [ threshold =",info[0],"] , [ accesses =",info[1],"]"
    
    print "Find TopK TA!!!"
    db0=create_lists(db)
    tt = time.time()
    info=TA(db0,2,k)
    tt = time.time() - tt 
    print "<TA>: [ threshold =",info[0],"] , [ accesses =",info[1],"] , [ tt = ", tt ," ]"
    
#     print "Find TopK PTA_2D!!!"
#     info=PTA_2D(partitioned_data_2D(db),2,k)
#     print "<PTA_2D>: [ threshold =",info[0],"] , [ accesses =",info[1],"] , [ tt = ", info[2] ," ]"
    
    print "Find TopK PTA!!!"
    info=PTA(db,2,k)
    print "<PTA>: [ threshold =",info[0],"] , [ accesses =",info[1],"] , [ tt = ", info[2] ," ]"

    