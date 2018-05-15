import sys


def gather(fname):
    fp = open(fname,"r")
    
    mm = dict()
    mm["tt_procesing"] = list()
    mm["tuple_count"] = list()
    
    for line in fp.readlines():
        if line.startswith("tt_procesing:"):
            data = line.strip().split(":")
            mm[data[0]].append(data[1])
        if line.startswith("tuple_count:"):
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
    
    for i in range(len(mm["tt_procesing"])):
        print mm["tt_procesing"][i],mm["tuple_count"][i]
    
    
    