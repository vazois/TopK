from itertools import combinations
from itertools import permutations

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
        self.part_num = (self.splits);
        
    def assign_to_part(self,stuple):
        tuple = [(a-1) for a in stuple]
        #tuple = stuple
        nom = math.fabs(sum([tuple[j] for j in range(self.d-1)]))
        denom = math.sqrt(self.d-1) * math.sqrt(sum([tuple[j]*tuple[j] for j in range(self.d)]))
        angle = math.degrees(math.asin(nom/denom))
        
        #print nom,denom,angle
        
        for id in range(len(self.bins)):
            if angle <= self.bins[id]:
                return id
        return self.part_num-1
    def __repr__(self):
        return repr((self.d,self.splits,self.split,self.bins,self.part_num))

class angle_part2:
    def __init__(self,d,splits):
        self.d = d
        self.splits = splits
        self.split = (math.pi/2)/self.splits
        self.bins = [(i+1)*self.split for i in range(self.splits)]
        self.part_num = (self.splits)**(self.d-1)
        
        if self.splits == 2:
            self.shf = 1
        elif self.splits == 4:
            self.shf = 2
        elif self.splits == 8:
            self.shf = 3
        elif self.splits == 16:
            self.shf = 4
        elif self.splits == 32:
            self.shf = 5
        elif self.splits == 64:
            self.shf = 6
            
    def hyperspherical_(self,tuple):
        stuple = [0 for j in range(self.d-1)]
        nom=0
        for j in range(self.d-1,1,-1):
            xn = tuple[j]
            xn_1 = tuple[j-1]
            nom+=xn*xn
            stuple[j-1] = math.sqrt(nom)/xn_1
        return stuple
    
    def assign_to_part(self,tuple):
        id = 0
        grid = []
        for i in range(len(tuple)):
            j = 0
            for j in range(len(self.bins)):
                if tuple[i] <= self.bins[j]:
                    id = (id | ( j<<self.shf ) )
                    break
        return id
        
    def __repr__(self):
        return repr((self.d,self.splits,self.split,self.bins,self.part_num))

class slope_tree:
    def __init__(self,d,slopes):
        self.d = d
        self.slopes = sorted(slopes,reverse=False)
        self.slope_num = len(slopes)
        self.cmb = [m for m in combinations([i for i in range(self.d)], 2)]
        self.part_num = (self.slope_num+1)*(self.d)
    
    def assign_to_part(self,tuple):
        id = 0
        for i in range(self.d-1,0,-1):
            j = 0
            for s in self.slopes:
                if  tuple[i] >= s*tuple[i-1]:
                    break
                j+=1
            id = (self.slope_num+1)*i + j

        return id
            
    def __repr__(self):
        return repr((self.d,self.slopes,self.slope_num))

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
    
    print "partitions:",len(data_parts)
    return [[len(data_parts[p]),d,data_parts[p]] for p in range(part_num) ]

def angle_partitioned_data2(db,splits):
    n = db[0]
    d = db[1]
    data = db[2]

    all_parts = [db[2]]
    for dd in range(d,1,-1):
        #print dd
        new_parts=[]
        for aps in all_parts:
            ap = angle_part(dd,splits)
            part_num = ap.part_num
            data_parts=[]
            
            for p in range(part_num):
                data_parts.append(list())
            
            for p in aps:
                id = ap.assign_to_part(p)
                data_parts[id].append(p)
            
            for part in data_parts:
                new_parts.append(part)
        all_parts=new_parts
        
    print "partitions:",len(all_parts)
    return [[len(p),d,p] for p in all_parts]

def angle_partitioned_data3(db,splits):
    n = db[0]
    d = db[1]
    data = db[2]
    
    ap = angle_part2(d,4)
#     print data[0]
#     print ap.hyperspherical_(data[0])
#     print ap.assign_to_part(ap.hyperspherical_(data[0]))
#     print "bins:",ap.bins
    print "part_num:",ap.part_num
    
    part_num = ap.part_num
    data_parts=[]
    for p in range(part_num):
        data_parts.append(list())
    
    for i in range(n):
        id = ap.assign_to_part(ap.hyperspherical_(data[i]))
        data_parts[id].append(data[i])
    
    for dp in data_parts:
        print "len:",len(dp)
    
    return [[len(data_parts[p]),d,data_parts[p]] for p in range(part_num) ]
    
        
        
    
