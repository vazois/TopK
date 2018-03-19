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
        self.split = (math.pi/2)/self.splits
        self.bins = [((i)*self.split) for i in range(0,self.splits)]
    
    def spherical(self,tuple):
        f=[m for m in range(self.d-1)]
        for i in range(self.d-1):
            nom = 0
            for j in range(i+1,self.d):
                nom+=tuple[j]*tuple[j]
            #print nom
            nom=math.sqrt(nom)
            f[i]=math.atan2(nom,tuple[i])
        return f
    
    def assign_to_part(self,tuple):
        id = 0
        stuple = self.spherical(tuple)
        for i in range(self.d-1):
            b = self.splits-1
            for j in range(self.splits):
                if stuple[i] <= self.bins[j]:
                    b=j
                    break
            id+=self.splits*i + b    
        return id
        
    def __repr__(self):
        return repr((self.d,self.splits,self.split,self.bins))

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

def bin_tree_partitioned_data(db):
    n = db[0]
    d = db[1]
    data = db[2]
    
    part_num = 2**(d-1)
#     if d > 2:
#         part_num = 2 ** d
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
            elif data[i][2] < data[i][1] and data[i][1] < data[i][0]:
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
    elif d == 5:
        for i in range(n):
            if data[i][4] >= data[i][3] and data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[0].append(data[i])
            elif data[i][4] >= data[i][3] and data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[1].append(data[i])
            elif data[i][4] >= data[i][3] and data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[2].append(data[i])
            elif data[i][4] >= data[i][3] and data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[3].append(data[i])
            elif data[i][4] >= data[i][3] and data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[4].append(data[i])
            elif data[i][4] >= data[i][3] and data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[5].append(data[i])
            elif data[i][4] >= data[i][3] and data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[6].append(data[i])
            elif data[i][4] >= data[i][3] and data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[7].append(data[i])
            elif data[i][4] < data[i][3] and data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[8].append(data[i])
            elif data[i][4] < data[i][3] and data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[9].append(data[i])
            elif data[i][4] < data[i][3] and data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[10].append(data[i])
            elif data[i][4] < data[i][3] and data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[11].append(data[i])
            elif data[i][4] < data[i][3] and data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[12].append(data[i])
            elif data[i][4] < data[i][3] and data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[13].append(data[i])
            elif data[i][4] < data[i][3] and data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[14].append(data[i])
            elif data[i][4] < data[i][3] and data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[15].append(data[i])
    elif d == 6:
        for i in range(n):
            if data[i][5] >= data[i][4] and data[i][4] >= data[i][3] and data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[0].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] >= data[i][3] and data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[1].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] >= data[i][3] and data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[2].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] >= data[i][3] and data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[3].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] >= data[i][3] and data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[4].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] >= data[i][3] and data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[5].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] >= data[i][3] and data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[6].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] >= data[i][3] and data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[7].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] < data[i][3] and data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[8].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] < data[i][3] and data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[9].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] < data[i][3] and data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[10].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] < data[i][3] and data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[11].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] < data[i][3] and data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[12].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] < data[i][3] and data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[13].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] < data[i][3] and data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[14].append(data[i])
            elif data[i][5] >= data[i][4] and data[i][4] < data[i][3] and data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[15].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] >= data[i][3] and data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[16].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] >= data[i][3] and data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[17].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] >= data[i][3] and data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[18].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] >= data[i][3] and data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[19].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] >= data[i][3] and data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[20].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] >= data[i][3] and data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[21].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] >= data[i][3] and data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[22].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] >= data[i][3] and data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[23].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] < data[i][3] and data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[24].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] < data[i][3] and data[i][3] >= data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[25].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] < data[i][3] and data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[26].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] < data[i][3] and data[i][3] >= data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[27].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] < data[i][3] and data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] >= data[i][0]:
                data_parts[28].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] < data[i][3] and data[i][3] < data[i][2] and data[i][2] >= data[i][1] and data[i][1] < data[i][0]:
                data_parts[29].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] < data[i][3] and data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] >= data[i][0]:
                data_parts[30].append(data[i])
            elif data[i][5] < data[i][4] and data[i][4] < data[i][3] and data[i][3] < data[i][2] and data[i][2] < data[i][1] and data[i][1] < data[i][0]:
                data_parts[31].append(data[i])    
            
                
        
    return [[len(data_parts[p]),d,data_parts[p]] for p in range(part_num) ]

def bin_tree_partitioned_data2(db):
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

def slope_tree_partitioned_data(db):
    n = db[0]
    d = db[1]
    data = db[2]
    
    st = slope_tree(d,[1])
    print "part_num:",st.part_num
    part_num = st.part_num
    data_parts=[]
    for p in range(part_num):
        data_parts.append(list())
    
    for i in range(n):
        id = st.assign_to_part(data[i])
        data_parts[id].append(data[i])
        
#     for part in data_parts:
#         print "slope part_size: ",len(part)
    
    return [[len(data_parts[p]),d,data_parts[p]] for p in range(part_num) ]

def spherical_partitioned_data(db,splits):
    n = db[0]
    d = db[1]
    data = db[2]
    
#     print "tuple: ",data[0]
    ap = angle_part(d,splits)
    f=ap.spherical(data[0])
#     print "f: ",f
#     print "bins:",ap.bins
#     print "id:",ap.assign_to_part(data[0])
    
    part_num = splits**(d-1)
    data_parts=[]
    for p in range(part_num):
        data_parts.append(list())
    
    for i in range(n):
        id = ap.assign_to_part(data[i])
        data_parts[id].append(data[i])
    
#     for i in range(part_num):
#         print i,":",len(data_parts[i])
    
    return [[len(data_parts[p]),d,data_parts[p]] for p in range(part_num) ]

    
    