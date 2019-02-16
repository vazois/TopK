import sys

inmemory_data = False

def gather(fname):
    global inmemory_data
    fp = open(fname,"r")
    
    mm = dict()
    mm["algo"]=list()
    mm["size"]=list()
    mm["threshold"]=list()
    mm["cpu_threshold"]=list()
    mm["gpu_threshold"]=list()
    mm["stop_level"]=list()
    mm["tt_procesing"] = list()
    mm["tuple_count"] = list()
    mm["accesses"] = list()
    mm["candidate_count"] = list()
    mm["tt_init"] = list()
    mm["tuples_per_second"] = list()
    
    for line in fp.readlines():
        if line.startswith("Generating"):
            inmemory_data = True
        if line.startswith("| Benchmark for"):
            data = line.strip().split(" ")
            #print data[4].split(",")[0][1:]+"-"+data[4].split(",")[2].split(")")[0]
            #mm["size"].append(data[4].split(",")[0][1:])
            mm["size"].append(data[4].split(",")[0][1:]+","+data[4].split(",")[2].split(")")[0])
            mm["algo"].append(data[3].strip())
        if line.startswith("threshold:"):
            data = line.strip().split(":")
            mm[data[0]].append(float(data[1]))
        if line.startswith("cpu_threshold:"):
            data = line.strip().split(":")
            mm[data[0]].append(float(data[1]))
        if line.startswith("gpu_threshold:"):
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
    
    if inmemory_data:
        print "Measurements Using Data Generated In Memory!!!"
    else:
        print "Measurements Using Data Loaded From File!!!"

    algo = ""
    sz = ""
    if len(mm["threshold"]) > 0:
        print "threshold",
    if len(mm["cpu_threshold"]) > 0:
        print "cpu_threshold",
    if len(mm["gpu_threshold"]) > 0:
        print "gpu_threshold",            
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

#    print "threshold tt_init tt_processing accesses tuple_count candidate_count"
    dt = list()
    for i in range(len(mm["algo"])):
        if(mm["algo"][i] != algo):
            algo = mm["algo"][i]
            print algo
        if(mm["size"][i] != sz):
            sz = mm["size"][i]
            print sz
        #print mm["tt_init"][i]
        #print mm["tt_procesing"][i],mm["tuple_count"][i],mm["tt_init"][i]
        #print mm["tuple_count"][i]
        #print mm["tuples_per_second"][i]
        #print "{0:0.4f}".format(mm["threshold"][i]),mm["stop_level"][i],"{0:0.4f}".format(mm["tt_init"][i]),"{0:0.4f}".format(mm["tt_procesing"][i]),mm["tuple_count"][i],mm["candidate_count"][i]
        #print "{0:0.4f}".format(mm["threshold"][i]),"{0:0.4f}".format(mm["tt_init"][i]),"{0:0.4f}".format(mm["tt_procesing"][i]),mm["tuple_count"][i],mm["candidate_count"][i]
        #print "{0:0.4f}".format(mm["threshold"][i]),"{0:0.4f}".format(mm["tt_init"][i]),"{0:0.4f}".format(mm["tt_procesing"][i]),mm["accesses"][i],mm["tuple_count"][i],mm["candidate_count"][i]
#         if i < len(mm["threshold"]):
#             print "{0:0.4f}".format(mm["threshold"][i]),
#         if i < len(mm["cpu_threshold"]):
#             print "{0:0.4f}".format(mm["cpu_threshold"][i]),
#         if i < len(mm["gpu_threshold"]):
#             print "{0:0.4f}".format(mm["gpu_threshold"][i]),                    
#         if i < len(mm["tt_init"]):
#             print "{0:0.4f}".format(mm["tt_init"][i]),
        if i < len(mm["tt_procesing"]):
            print "{0:0.4f}".format(mm["tt_procesing"][i]),
        if i < len(mm["accesses"]):
            print mm["accesses"][i],
        if i < len(mm["tuple_count"]):
            print mm["tuple_count"][i],
        if i < len(mm["candidate_count"]):
            print mm["candidate_count"][i],        
        print ""
    
#     for i in range(len(mm["algo"])):
#         if(mm["algo"][i] != algo):
#             algo = mm["algo"][i]
#             print algo
#         if(mm["size"][i] != sz):
#             dt = list()
#             sz = mm["size"][i]
#             print sz
#         if i < len(mm["tt_procesing"]):
#             dt.mm["tt_procesing"][i]
            
            
            
            
            
                       
        