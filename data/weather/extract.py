import sys
from subprocess import call
from subprocess import check_output

def extract_csv(fname):
    station = dict()
    
    print "Loading file ... "
    rows=int(check_output(["wc","-l",fname]).split(" ")[0])
    step = (rows/100)
    with open(fname,'r') as fp:
        ll = 0
        for line in fp:
            data = line.strip().split(",")
            
            if data[2] == "TMAX" or data[2] == "TMIN":
                sid = data[0]
                #print sid
                if sid not in station:
                    station[sid] = dict()
                
                type = data[2]
                if type not in station[sid]:
                    station[sid][type] = dict()
                
                date = data[1][4:]
                if date not in station[sid][type]:
                    station[sid][type][date] = list()
                station[sid][type][date].append(float(data[3])/10)
            
            if ll % step == 0:
                sys.stdout.write("\r%s%%" % str(round((float(ll)/rows)*100)))
                sys.stdout.flush()
            ll+=1
    print "Finish loading file ... "
    return station

def gather_stats(station):
    global attributes
    
    count_attr = dict()
    for s in station:
        for t in station[s]:
            for d in station[s][t]:
                size = len(station[s][t][d])
                if size not in count_attr:
                    count_attr[size]=0
                count_attr[size]+=1
    for ca in count_attr:
        print ca,":",count_attr[ca]

def write_csv(fname,station):
    global attributes
    fp = open(fname.split(".")[0],'w')
    count = 0
    for s in station:
        for t in station[s]:
            for d in station[s][t]:
                if len(station[s][t][d]) >= attributes:
                    size = len(station[s][t][d])
                    for v in station[s][t][d]:
                        fp.write(str(v)+",")
                    count+=1
    fp.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage:",sys.argv[0],"<file> <attributes>"
        exit(1)
    
    fname = sys.argv[1]
    attributes = int(sys.argv[2])
    print "Process file:", fname, attributes
    station=extract_csv(fname)
    gather_stats(station)
