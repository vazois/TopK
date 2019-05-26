import os
import sys
from subprocess import call
from subprocess import check_output

OUTPUT="output.csv"
LIMIT=1*1024*1024
VARS=["TMAX","TMIN","WSF2","WSF5"]
SIMILAR_TYPE_GROUPING=True

def extract_csv(fname,d):
	global OUTPUT,LIMIT, factor
	outp = open("output.csv","w")
	cdata=dict()
	rows=0

	n = int(fname.split("_")[1])
	step=int(float(n)/100)
	i=0
	j=0
	with open(fname,'r') as fp:
		for line in fp:
			data = line.strip().split(",")
			sid = data[0]
			#sid = sid[0:sid.find('0')]
			if sid not in cdata:
				cdata[sid]=list()
			if (float(data[2])) >= -50 and (float(data[2])) < 50:
			#if ((float(data[2])) >= -100 and (float(data[2])) < -50) or ((float(data[2])) >= 50 and (float(data[2])) < 100):
				cdata[sid].append(float(data[2])/10)
			if len(cdata[sid]) == d:
				line=""
				for v in cdata[sid]:
					line+=str(v)+","
				outp.write(line[:-1]+"\n")
				cdata[sid]=list()
				rows+=1
				#if LIMIT < rows:
				#	outp.close()
				#	return rows
			
			if i % step == 0:
				sys.stdout.write("\r%s%%" % str(round((float(i)/n)*100)))
				sys.stdout.flush()
				j+=1
			i+=1
			if rows > (factor*1024*1024 - 1):
				break
	outp.close()
	return rows

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print "Usage:",sys.argv[0],"<input_file> <dimension> <data multiplication factor>"
		exit(1)
    
	fname = sys.argv[1]
	d = int(sys.argv[2])
	factor = int(sys.argv[3])
	print "Extract csv data:", fname
	rows=extract_csv(fname,d)
	print ""
	print rows,d
	os.rename(OUTPUT,"wtuples_"+str(rows)+"_"+str(d))