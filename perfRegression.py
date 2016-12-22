import sys

if ( len(sys.argv) != 3):
    print "USAGE: <old file> <new file>\n"
    exit(-1)

fileOld = open(sys.argv[1], "r")
fileNew = open(sys.argv[2], "r")

old = []
for l in fileOld:
    old.append(float(l.split()[-2]))
new = []
for l in fileNew:
    new.append(float(l.split()[-2]))

avgSpeedup = 0
for i in range(len(old)):
    speedup = new[i]/old[i]
    avgSpeedup += speedup
    print i, speedup

print "Avg Speedup: ", avgSpeedup / len(old)
