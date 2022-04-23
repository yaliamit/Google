import subprocess
import time
import sys
import os
time_format="%Y-%m-%dT%H:%M:%S+0000"



if sys.platform == 'linux':
    predir=''
else:
    predir='../'


f = open(predir+'raspi.txt', 'r')
ip_nos=[]
cli=[]
room=[]
for l in f.readlines():
        dat = l.split(';')
        ip_nos+=[dat[1]]
        cli+=[dat[2]]
        room+=[dat[0]]
f.close()

f=open('check_report.txt','a')
time_format="%Y-%m-%dT%H:%M:%S+0000"
stime=time.strftime(time_format+'\n')
f.write(stime)
num_clients=len(ip_nos)
good=0
for ip,cl,rom in zip(ip_nos,cli,room):
    ad='pi@'+ip
    cmd=['ssh',ad,'ps -C python3 |  wc -l']
    f.write(' '.join(cmd)+'\n')
    res=subprocess.run(cmd,stdout=subprocess.PIPE)
    if len(res.stdout)>0:
        rr=int(res.stdout[:-1])
    else:
        rr=0
    f.write('rr:'+str(rr)+'\n')
    if rr==2:
        good+=1
        f.write(rom+' good'+'\n')
    else: # Try reruning
        f.write('Rerunning '+str(rom)+'\n')
        cmd = 'scp rerun'+str(int(cl))+'.sh '+ad+':.'
        f.write(cmd+'\n')
        os.system(cmd)
        cmd = 'ssh '+ad+ ' sh rerun'+str(int(cl))+'.sh &'
        f.write(cmd+'\n')
        os.system(cmd)
        time.sleep(15)
        cmd = ['ssh', ad, 'ps -C python3 | wc -l']
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        if len(res.stdout) > 0:
            rr = int(res.stdout[:-1])
        else:
            ff=open('junk.txt','w')
            ff.write('Please unplug and replug' + rom+'\n')
            ff.close()
            cmd = 'cat junk.txt | mail -s "Restart" patrickstycos@gmail.com'
            os.system(cmd)
            os.system("rm junk.txt")
            rr = 0
        f.write('rr:' + str(rr)+'\n')
        if rr == 2:
            good += 1
            f.write(rom + 'good'+'\n')

f.write('good clients '+ str(good)+' ' +str(num_clients)+'\n')
#If some processes aren't running restart everything
# if good<num_clients-2:
#     cmd=['kill_all.sh']
#     res=subprocess.run(cmd,stdout=subprocess.PIPE)
#     cmd=['./dump_patrick.sh']
#     res=subprocess.run(cmd,stdout=subprocess.PIPE)
#     cmd=['rm patricks_dump.pkl']
#     res=subprocess.run(cmd,stdout=subprocess.PIPE)
#     cmd=['python3 run_all.py &']
#     res=subprocess.run(cmd,stdout=subprocess.PIPE)
#
