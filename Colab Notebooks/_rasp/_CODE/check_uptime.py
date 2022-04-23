import subprocess
import sys
import os

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

num_clients=len(ip_nos)
for ip,cl,rom in zip(ip_nos,cli,room):
    ad='pi@'+ip
    #cmd=['ssh',ad,'crontab -l']
    #print(cmd)
    #res=subprocess.run(cmd,stdout=subprocess.PIPE)
    #print(res.stdout)
    #comm1 = 'scp rerun.sh ' + ad + ':.'
    #print(comm1)
    #os.system(comm1)
    #cmd=['ssh',ad,'pip list | grep socketio']
    #res = subprocess.run(cmd, stdout=subprocess.PIPE)
    #print(res.stdout)
    cmd = ['ssh', ad, ' uptime -s ']
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    print(ad,res.stdout)

