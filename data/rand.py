import sys
import random


def next_value(distr):
    if (distr == "u"):
        return random.uniform(0.1,1.0)
    

if(len(sys.argv) < 3):
     print("Usage: " + sys.argv[0] + " <cardinality> <attributes> <distribution: (u)niform>")
     exit()
     
n = int(sys.argv[1])
d = int(sys.argv[2])
distr = "u"
if (len(sys.argv) == 4):
    distr=sys.argv[3]
    

#print next_value(distr)

fname= "d_"+str(n)+"_"+str(d)+"_"+distr
fp = open(fname,'w')
for i in range(n):
    line = ""
    for j in range(d):
        line+=str(next_value(distr))+","
    line=line[:-1]
    #print line
    fp.write(line+"\n")
    
fp.close()