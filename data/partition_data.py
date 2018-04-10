from itertools import combinations
from itertools import permutations

from numpy import median
import numpy as np

import math

class tuple:
    def __init__(self, id, score):
        self.id = id
        self.score = score
    def __repr__(self):
        return repr((self.id, self.score))

class bin_tree:
    def __init__(self,d):
        self.d = d
    def assign_to_part(self,tuple):
        id = 0
        for i in range(self.d-1,0,-1):
            if  tuple[i] < tuple[i-1]:#x0>=x1 x1>=x2 x2>=x3
                id = ( id | (0x1 << (i-1)) )
        return id
        
    def __repr__(self):
        return repr((self.d))
  
class angle_part:
    def __init__(self,d,splits):
        self.d = d
        self.splits = splits
        self.split = 90.0/self.splits
        self.bins = [(i+1)*self.split for i in range(self.splits)]
        self.part_num = (self.splits**(self.d-1))
        self.shf = int(math.log(self.splits,2))
        
    def assign_to_part(self,stuple):
        tuple = [(a-2) for a in stuple]
        #tuple = stuple
#         nom = math.fabs(sum([tuple[j] for j in range(self.d-1)]))
#         denom = math.sqrt(self.d-1) * math.sqrt(sum([tuple[j]*tuple[j] for j in range(self.d)]))
#         angle = math.degrees(math.asin(nom/denom))
#         
#         for id in range(len(self.bins)):
#             if angle <= self.bins[id]:
#                 return id
#         
#         return self.part_num-1
        polar = [ ]
        sum = 0
        #print tuple,stuple
        for i in range(self.d-1,0,-1):
            sum+=tuple[i]*tuple[i]
            tanf= math.sqrt(float(sum))/tuple[i-1]
            polar.append(math.fabs(math.degrees(math.atan(tanf))))
        
        id = 0
        for i in range(len(polar)):
            b = -1
            for bin in self.bins:
                if polar[i] <= bin:
                    break
                b+=1
            id = id | (b << (i*self.shf))
        #print id,self.part_num
        #print polar, stuple, self.bins
        return id
     
    def __repr__(self):
        return repr((self.d,self.splits,self.split,self.bins,self.part_num))

class angle_part2:
    def __init__(self,d,splits):
        self.d = d
        self.splits = splits
        self.bins = []
        self.part_num = (self.splits**(self.d-1))
        self.shf = int(math.log(self.splits,2))
            
    def polar_(self,stuple):
        tuple = [(a-1) for a in stuple]
        polar = [0]*(self.d-1)
        sum = 0
        #print tuple,stuple
        for i in range(self.d-1,0,-1):
            sum+=tuple[i]*tuple[i]
            tanf= math.sqrt(float(sum))/tuple[i-1]
            #polar.append(math.fabs(math.degrees(math.atan(tanf))))
            polar[i-1]=(math.fabs(math.degrees(math.atan(tanf))))
        
#         for i in range(0,self.d-1):
#             sum+=tuple[i+1]*tuple[i+1]
#             tanf= math.sqrt(float(sum))/tuple[i]
#             polar.append(math.fabs(math.degrees(math.atan(tanf))))
        
        return polar
    
    def find_splits(self,pdata):
        split = float(len(pdata[0]))/self.splits
        for j in range(len(pdata)):
            k = int(split)
            c = pdata[j]
            bb = []
            for i in range(self.splits-1):
                idx = np.argpartition(c,k)
                v = c[idx[k-1]]
                bb.append(v)
                k+=int(split)
            self.bins.append(bb)
        #self.part_num = len(self.bins)*(len(self.bins[0])+1)
        for b in self.bins:
            print b
        
    def assign_to_part(self,tuple):
        x=0
        id = 0
        polar = self.polar_(tuple)
        for i in range(len(polar)):
            b = 0
            for j in range(len(self.bins[i])):
                if polar[i] <= self.bins[i][j]:
                    break
                b+=1
            b = self.splits-1 if b == self.splits else b
            id = id | (b << (i*self.shf))
            #print polar,b,id
        return id
        
    def __repr__(self):
        return repr((self.d,self.splits,self.split,self.bins,self.part_num))
    
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

def random_partitioned_data(db,splits):
    n = db[0]
    d = db[1]
    data = db[2]
    
    part_num = splits
    data_parts=[]
    for p in range(part_num):
        data_parts.append(list())
        
    for i in range(n):
        id = i % splits
        data_parts[id].append(data[i])
    
    return [[len(data_parts[p]),d,data_parts[p]] for p in range(part_num) ]

def bin_tree_partitioned_data(db):
    n = db[0]
    d = db[1]
    data = db[2]
    
    part_num = 2**(d-1)
    data_parts=[]
    for p in range(part_num):
        data_parts.append(list())
    
    bt = bin_tree(d)
    for i in range(n):
        id = bt.assign_to_part(data[i])
        data_parts[id].append(data[i])
    
    return [[len(data_parts[p]),d,data_parts[p]] for p in range(part_num) ]

def bin_tree_partitioned_data2(db):
    n = db[0]
    d = db[1]
    data = db[2]
    
    all_parts = [db[2]]
    #for dd in range(2,d,1):
    for dd in range(d-1,1,-1):
        new_parts=[]
        bt = bin_tree(dd)
        for aps in all_parts:
            part_num = 2**(dd-1)
            data_parts=[list() for p in range(part_num)]
            
            for p in aps:
                id = bt.assign_to_part(p)
                data_parts[id].append(p)
            
            for part in data_parts:
                new_parts.append(part)
        all_parts=new_parts
    
    print "Part_num:",len(all_parts)
    return [[len(p),d,p] for p in all_parts]

def angle_partitioned_data(db,splits):
    n = db[0]
    d = db[1]
    data = db[2]
    
    ap = angle_part(d,splits)    
    part_num = ap.part_num
    data_parts=[]
    
    for p in range(part_num):
        data_parts.append(list())
    
    for i in range(n):
        id = ap.assign_to_part(data[i])
        data_parts[id].append(data[i])
    
    #print "partitions:",len(data_parts)
    return [[len(data_parts[p]),d,data_parts[p]] for p in range(part_num) ]

def angle_partitioned_data2(db,splits):
    n = db[0]
    d = db[1]
    data = db[2]
    
    #print "Angle Partitioning 2!!!"
    
    ap = angle_part2(d,splits)
    pdata = [[] for i in range(d-1)]
    m = 0
    for tuple in data:
        polar = ap.polar_(tuple)
        
        for i in range(len(polar)):
            pdata[i].append(polar[i])
    
    ap.find_splits(pdata)
    pdata=[]
    part_num = ap.part_num
    data_parts=[]
    for p in range(part_num):
        data_parts.append(list())
        
    for i in range(n):
        id = ap.assign_to_part(data[i])
        if( i  < 16 ):
            print ["{0:.4f}".format(a) for a in data[i]],["{0:.4f}".format(a) for a in ap.polar_(data[i])], '%02d' % id
        #print id,len(data_parts)
        data_parts[id].append(data[i])
    
#     for i in range(len(data_parts)):
#         print i,len(data_parts[i])
    
    return [[len(data_parts[p]),d,data_parts[p]] for p in range(part_num) ]
