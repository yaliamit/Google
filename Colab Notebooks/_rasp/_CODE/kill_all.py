import os
import time

f = open('raspi.txt', 'r')

for l in f.readlines():
    l=l.replace('\n','')
    dat = l.split(';')
    comm1 = 'ssh pi@' + dat[1] + ' killall python3'
    print(comm1)
    os.system(comm1)

f.close()
