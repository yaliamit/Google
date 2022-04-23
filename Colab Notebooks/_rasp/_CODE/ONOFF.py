import requests
import os
import datetime as dt
f=open('boiler_threshes.txt','r')
a=f.readlines()
f.close()
aa=a[0].strip('\n').split(':')
boiler_thresh_low=float(aa[0])
boiler_thresh_high=float(aa[1])
tim=dt.datetime.now()


r = requests.post('http://192.168.0.174/cm?cmnd=Power')
u=r.json()
on=True
if (on):
    try:
        r = requests.post('http://192.168.0.174/cm?cmnd=Power%20on')
        u = r.json()

    except requests.exceptions.RequestException as e:
        print('Was not able to turn on the boiler at time ' + str(tim))
        ff = open('junk.txt', 'w')
        ff.write('Was not able to turn on the boiler at time ' + str(tim) + '\n')
        ff.close()
        cmd = 'cat junk.txt | mail -s "Boiler on fail" amityali@gmail.com'
        #os.system(cmd)
        os.system("rm junk.txt")
else:
    try:
        r = requests.post('http://192.168.0.174/cm?cmnd=Power%20off')
        u = r.json()
        print(r)
    except requests.exceptions.RequestException as e:
        print('Was not able to turn off the boiler at time', tim)
        ff = open('junk.txt', 'w')
        ff.write('Was not able to turn off the boiler at time' + + '\n')
        ff.close()
        #cmd = 'cat junk.txt | mail -s "Boiler off fail" amityali@gmail.com'
        os.system(cmd)
        os.system("rm junk.txt")