import os
import subprocess

ad='pi@192.168.0.199'

cmd=['ssh',ad,'ls /var/www/laundry/today* | wc -l']
res=subprocess.run(cmd,stdout=subprocess.PIPE)

num=int(res.stdout)
startnum=num
if num==7:
    startnum=6

for i in range(startnum,0,-1):
    #print(i)
    cmd='ssh '+ad+' mv /var/www/laundry/today_dump_'+str(i)+'.png /var/www/laundry/today_dump_'+str(i+1)+'.png'
    os.system(cmd)

cmd='scp today_dump.png '+ad+':/var/www/laundry/today_dump_1.png'
os.system(cmd)
