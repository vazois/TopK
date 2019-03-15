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
    if len(mm["threshold"]) > 0:
        print "threshold",
    if len(mm["tt_init"]) > 0:
        print "tt_init",
    if len(mm["tt_procesing"]) > 0:
        print "tt_procesing",
    if len(mm["accesses"]) > 0:
        print "accesses",
    if len(mm["tuple_count"]):
        print "tuple_count",
    if len(mm["candidate_count"]) > 0:
        print "candidate_count",
    print "" 

    pp = False
#    print "threshold tt_init tt_processing accesses tuple_count candidate_count"
    for i in range(len(mm["tt_init"])):
        if(mm["algo"][i] != algo):
            algo = mm["algo"][i]
            #print algo
            pp = True
        #if algo != "DL":
        #    continue
        if pp:
            print algo
            pp = False
        #print mm["tt_init"][i]
        #print mm["tt_procesing"][i],mm["tuple_count"][i],mm["tt_init"][i]
        #print mm["tuple_count"][i]
        #print mm["tuples_per_second"][i]
        #print "{0:0.4f}".format(mm["threshold"][i]),mm["stop_level"][i],"{0:0.4f}".format(mm["tt_init"][i]),"{0:0.4f}".format(mm["tt_procesing"][i]),mm["tuple_count"][i],mm["candidate_count"][i]
        #print "{0:0.4f}".format(mm["threshold"][i]),"{0:0.4f}".format(mm["tt_init"][i]),"{0:0.4f}".format(mm["tt_procesing"][i]),mm["tuple_count"][i],mm["candidate_count"][i]
        #print "{0:0.4f}".format(mm["threshold"][i]),"{0:0.4f}".format(mm["tt_init"][i]),"{0:0.4f}".format(mm["tt_procesing"][i]),mm["accesses"][i],mm["tuple_count"][i],mm["candidate_count"][i]
        if i < len(mm["threshold"]):
            print "{0:0.4f}".format(mm["threshold"][i]),
        if i < len(mm["tt_init"]):
            print "{0:0.4f}".format(mm["tt_init"][i]),
        if i < len(mm["tt_procesing"]):
            print "{0:0.4f}".format(mm["tt_procesing"][i]),
        if i < len(mm["accesses"]):
            print mm["accesses"][i],
        if i < len(mm["tuple_count"]):
            print mm["tuple_count"][i],
        if i < len(mm["candidate_count"]):
            print mm["candidate_count"][i],        
        print ""

    
    
    