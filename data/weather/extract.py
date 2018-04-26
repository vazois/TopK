import os
import sys
from subprocess import call
from subprocess import check_output

OUTPUT="output.csv"

def write_out(fp,values,tuples):
	output_row =""
	for v in values:
		output_row+=str(v)+","
	if tuples==0:
		fp.write(output_row[0:-1])
	else:
		fp.write("\n"+output_row[0:-1])

def build_csv(fname):
	global attributes, OUTPUT
	station = dict()
    
	print "Processing file (",fname,") ... "
	rows=int(check_output(["wc","-l",fname]).split(" ")[0])
	step = (rows/100)
	out_fp = open(OUTPUT,'w')
	tuples = 0
	with open(fname,'r') as fp:
		ll = 0
		for line in fp:
			data = line.strip().split(",")
            
			sid = data[0]
			if sid not in station:
				station[sid] = dict()
                
			type = data[2]
			if type not in station[sid]:
				station[sid][type] = dict()
			
			date = data[1][4:]
			if date not in station[sid][type]:
				station[sid][type][date] = list()
			
			station[sid][type][date].append(float(data[3])/10)
		
			if len(station[sid][type][date]) == attributes:
				write_out(out_fp,station[sid][type][date],tuples)
				station[sid][type][date] = list()
				tuples+=1
				
			if ll % step == 0:
				sys.stdout.write("\r%s%%" % str(round((float(ll)/rows)*100)))
				sys.stdout.flush()
			ll+=1
	out_fp.close()
	os.rename(OUTPUT,"weather_"+str(tuples)+"_"+str(attributes))
	print "Finish processing file(",fname,") ... "

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage:",sys.argv[0],"<file> <attributes>"
        exit(1)
    
    fname = sys.argv[1]
    attributes = int(sys.argv[2])
    print "Process file:", fname, attributes
    station=build_csv(fname)
