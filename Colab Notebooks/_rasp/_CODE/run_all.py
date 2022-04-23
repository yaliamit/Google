import os
import time
os.system('python3 rp_server.py > out &')

time.sleep(3)

f = open('raspi.txt', 'r')

for l in f.readlines():
    l=l.replace('\n','')
    dat = l.split(';')
    comm1 = 'scp client' + dat[2] + '.py pi@' + dat[1] + ':.'
    print(comm1)
    os.system(comm1)
    comm1 = 'scp rerun'+dat[2]+'.sh pi@' + dat[1] + ':.'
    print(comm1)
    os.system(comm1)
    comm2='ssh pi@'+dat[1]+' "python3 client'+dat[2]+'.py >> out &" &'
    print(comm2)
    os.system(comm2)
    
f.close()
