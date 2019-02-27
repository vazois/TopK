import sys

def trace_file(fname):
    fp = open(fname,'r')
    
    mm = dict()
    
    for line in fp.readlines():
        #print line
        if "gpta_atm_16" in line:
            data = line.strip().split()
            #print data[3]
            mm["gpta_atm_16"]= data[3].split("ms")[0]
    fp.close()
    return mm

def metrics_file(fname):
    fp = open(fname,'r')
    
    mm = dict()
    mm = list()
    for line in fp.readlines():
        if "dram_read_throughput" in line:
            data = line.strip().split()
            #print data[8].split("GB/s")[0]
            mm.append(data[8].split("GB/s")[0])
        if "achieved_occupancy" in line:
            data = line.strip().split()
            #print data[6]
            mm.append(data[6])
    
    fp.close()
    return mm  

def print_(input,tr,mr):
    
    for algo in tr:
        print algo,": [",input,"]"
        out = tr[algo]
        for m in mr:
            #print m
            out += " " + m
        print out

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage:",sys.argv[0],"<trace> <metrics>"
        exit(1)
    
    trace = sys.argv[1]
    #print "Process trace: ", trace
    if "trace" not in trace:
        print trace, " not a trace file!!!"
        exit(1)
    metrics = sys.argv[2]
    #print "Process metrics: ", metrics
    if "metrics" not in metrics:
        print metrics, " not a metric file!!!"
        exit(1)
        
    #trace_file(trace)
    #metrics_file(metrics)
    print_(trace.split("_trace.log")[0],trace_file(trace),metrics_file(metrics))
    
    
    
    
    