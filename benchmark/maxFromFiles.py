import sys
from shutil import copyfile
import os.path


f1Name = sys.argv[1]
f2Name = sys.argv[2]
columNum = 7

if( not os.path.isfile(f1Name) ):
   exit(0)
if( not os.path.isfile(f2Name) ):
   copyfile(f1Name, f2Name)
   exit(0)

f1 = open(f1Name)
f2 = open(f2Name)

f1Content = []
for l in f1:
   f1Content.append(l)
f2Content = []
for l in f2:
   f2Content.append(l)

f1.close()
f2.close()
os.remove(f1Name)

#create new file
f2 = open(f2Name,"w")
for i in range(len(f1Content)):
   try:
      if( float(f1Content[i].split()[columNum ]) > float(f2Content[i].split()[columNum ]) ):
         f2.write(f1Content[i])
      else:
         f2.write(f2Content[i])
   except:
      print "ERROR:", i
f2.close()
