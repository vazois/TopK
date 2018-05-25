import sys


def gather(fname):
    fp = open(fname,"r")
    
    mm = dict()
    mm["tt_procesing"] = list()
    mm["tuple_count"] = list()
    mm["tt_init"] = list()
    #mm["tuples_per_second"] = list()
    
    for line in fp.readlines():
        if line.startswith("tt_procesing:"):
            data = line.strip().split(":")
            mm[data[0]].append(data[1])
        if line.startswith("tuple_count:"):
            data = line.strip().split(":")
            mm[data[0]].append(data[1])
        if line.startswith("tt_init:"):
            data = line.strip().split(":")
            mm[data[0]].append(data[1])
        if line.startswith("tuples_per_second:"):
            data = line.strip().split(":")
            mm[data[0]].append(data[1])
    
    fp.close()
    
    return mm

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage:",sys.argv[0],"<file>"
        exit(1)
    
    fname = sys.argv[1]
    print "Process file: ", fname
    mm = gather(fname)
    
#     for measurement in mm:
#         print measurement
#         for v in mm[measurement]:
#             print v
    
    for i in range(len(mm["tt_init"])):
        #print mm["tt_init"][i]
        #print mm["tt_procesing"][i],mm["tuple_count"][i],mm["tt_init"][i]
        print mm["tuple_count"][i]
        #print mm["tuples_per_second"][i]
        #print mm["tt_procesing"][i]

    
    
    