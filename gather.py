import sys


def gather(fname):
    fp = open(fname,"r")
    
    mm = dict()
    mm["algo"]=list()
    mm["threshold"]=list()
    mm["stop_level"]=list()
    mm["tt_procesing"] = list()
    mm["tuple_count"] = list()
    mm["accesses"] = list()
    mm["candidate_count"] = list()
    mm["tt_init"] = list()
    mm["tuples_per_second"] = list()
    
    for line in fp.readlines():
        if line.startswith("< Benchmark for"):
            data = line.strip().split(" ")
            #print data[3]
            mm["algo"].append(data[3])
        if line.startswith("threshold:"):
            data = line.strip().split(":")
            mm[data[0]].append(float(data[1]))
        if line.startswith("stop_level:"):
            data = line.strip().split(":")
            mm[data[0]].append(float(data[1]))
        if line.startswith("tt_procesing:"):
            data = line.strip().split(":")
            mm[data[0]].append(float(data[1]))
        if line.startswith("candidate_count:"):
            data = line.strip().split(":")
            mm[data[0]].append(float(data[1]))
        if line.startswith("tuple_count:"):
            data = line.strip().split(":")
            mm[data[0]].append(data[1].strip())
        if line.startswith("accesses:"):
            data = line.strip().split(":")
            mm[data[0]].append(data[1].strip())
        if line.startswith("tt_init:"):
            data = line.strip().split(":")
            mm[data[0]].append(float(data[1]))
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
    
    algo = ""
    print "threshold tt_init tt_processing accesses tuple_count candidate_count"
    for i in range(len(mm["tt_init"])):
        if(mm["algo"][i] != algo):
            algo = mm["algo"][i]
            print algo
        #print mm["tt_init"][i]
        #print mm["tt_procesing"][i],mm["tuple_count"][i],mm["tt_init"][i]
        #print mm["tuple_count"][i]
        #print mm["tuples_per_second"][i]
        #print "{0:0.4f}".format(mm["threshold"][i]),mm["stop_level"][i],"{0:0.4f}".format(mm["tt_init"][i]),"{0:0.4f}".format(mm["tt_procesing"][i]),mm["tuple_count"][i],mm["candidate_count"][i]
        #print "{0:0.4f}".format(mm["threshold"][i]),"{0:0.4f}".format(mm["tt_init"][i]),"{0:0.4f}".format(mm["tt_procesing"][i]),mm["tuple_count"][i],mm["candidate_count"][i]
        print "{0:0.4f}".format(mm["threshold"][i]),"{0:0.4f}".format(mm["tt_init"][i]),"{0:0.4f}".format(mm["tt_procesing"][i]),mm["accesses"][i],mm["tuple_count"][i],mm["candidate_count"][i]

    
    
    