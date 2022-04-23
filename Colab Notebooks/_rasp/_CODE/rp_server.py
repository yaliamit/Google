#I don't really understand these two packages.
#Must use python 3 or higher
#much of the code here came from youtube.com/watch?v=ZEsulQgsORc
import eventlet
import socketio
import time
import os
import pickle
import sys
import requests
import datetime as dt


time_format="%Y-%m-%d %H:%M:%S.%f"
data_all={}
n = {'n0':dt.datetime.now(),'n1':dt.datetime.now(),'n2':0}
boiler_thresh_low=68
boiler_thresh_high=71
global t1
t1=time.time()

def control_boiler(tim):

    with open('boiler_threshes.txt', 'r') as f:
        a = f.readlines()

    aa = a[0].strip('\n').split(':')
    boiler_thresh_low_day = float(aa[0])
    boiler_thresh_high_day = float(aa[1])
    boiler_thresh_low_night = float(aa[2])
    boiler_thresh_high_night = float(aa[3])
    out_thresh=float(aa[4])
    wd=tim.weekday()
    print('DAY',wd,tim.hour,tim.minute)
    if wd==5 or wd==6:
      if tim.hour >= 7 and tim.hour <=23:
         boiler_thresh_low=boiler_thresh_low_day
         boiler_thresh_high=boiler_thresh_high_day
      else:
         boiler_thresh_low = boiler_thresh_low_night
         boiler_thresh_high = boiler_thresh_high_night
    else:
      if (tim.hour >= 5 and tim.hour <=22) or (tim.hour>22 and tim.hour<=23 and tim.minute<30):
         boiler_thresh_low=boiler_thresh_low_day
         boiler_thresh_high=boiler_thresh_high_day
      else:
         boiler_thresh_low = boiler_thresh_low_night
         boiler_thresh_high = boiler_thresh_high_night

    average=0
    num=0
    outside=None

    for keys, values in data_all.items():
        if 'out' not in keys:
            if values==[]:
                print('Not all values are there',keys)
            else:
                print(keys, values[-1])
                vtime=dt.datetime.strptime(values[-1][0],time_format)
                if (tim-vtime).seconds <= 300:
                    average+=values[-1][1]
                    num+=1
        elif 'temp' in keys:
            outside=float(values[0][1])
    if num >= 9:
        average/=num
    else:
        print("Not enough thermometers to compute average")
        return

    flag=True
    if average <= boiler_thresh_low:
        try:
            r = requests.post('http://192.168.0.174/cm?cmnd=Power')
            u = r.json()
            if (u['POWER'] == 'OFF'):
                r=requests.post('http://192.168.0.174/cm?cmnd=Power%20on')
                with open('boiler_record.csv','a') as f:
                    f.write(str(tim)+','+str(average)+',' + str(boiler_thresh_low) + ',' + str(boiler_thresh_high)+',ON\n')
                    flag=False

        except requests.exceptions.RequestException as e:
            print('Was not able to turn on the boiler at time '+str(tim))
            ff = open('boiler_record.txt', 'a')
            ff.write('Was not able to turn on the boiler at time '+str(tim) + '\n')
            ff.close()
            cmd = 'cat junk.txt | mail -s "Boiler on fail" patrickstycos@gmail.com'
            os.system(cmd)
            os.system("rm junk.txt")

    elif average >= boiler_thresh_high or (outside is not None and outside>out_thresh):
        try:
            r = requests.post('http://192.168.0.174/cm?cmnd=Power')
            u=r.json()
            if (u['POWER']=='ON'):
                r = requests.post('http://192.168.0.174/cm?cmnd=Power%20off')
                with open('boiler_record.csv', 'a') as f:
                    f.write(str(tim)+','+str(average)+',' + str(boiler_thresh_low) + ',' + str(boiler_thresh_high)+',OFF\n')
                    flag=False

        except requests.exceptions.RequestException as e:
            print('Was not able to turn off the boiler at time',tim)
            ff = open('boiler.txt', 'w')
            ff.write('Was not able to turn off the boiler at time' + + '\n')
            ff.close()
            cmd = 'cat junk.txt | mail -s "Boiler off fail" patrickstycos@gmail.com'
            os.system(cmd)
            os.system("rm junk.txt")
    #We haven't written a switch ON/OFF to record. Just write average.
    if flag:
        with open('boiler_record.csv', 'a') as f:
            f.write(str(tim) + ',' + str(average) + ',' + str(boiler_thresh_low) + ',' + str(boiler_thresh_high)+',None\n')

    sys.stdout.flush()
#creates the server instance
sio = socketio.Server()

#no idea... but it's used in the final line of the code
app = socketio.WSGIApp(sio)

#Every time a connection is made The below event will happen
#Use to create a log of all connection
@sio.event
def connect(sid, environ):
 
    print('connect ', sid)

#every time a message is recieved from a client the following code runs
@sio.event
def my_message(sid, data):
    data_all[data['loc']]+=[[data['ts'],data['temp']]]
    n['n2']=dt.datetime.now()
    #print('top',n)
    if (n['n2']-n['n0']).seconds>=60:
        control_boiler(n['n2'])
        n['n0']=n['n2']
        print('boiler',n)
    if (n['n2']-n['n1']).seconds>=1200:
        r = requests.post('https://api.openweathermap.org/data/2.5/onecall?lat=41.92357399688181&lon=-87.7062108571584&appid=0087fed856bfa781fb30543e395a1fc4')
        u=r.json()
        weather=[u['current']['temp']*9/5-459.67, u['current']['wind_speed']]
        data_all['out_temp']=[str(dt.datetime.now()),weather[0]]
        data_all['out_wind']=[str(dt.datetime.now()),weather[1]]
        n['n1']=n['n2']
        f=open('patricks_dump.pkl','a+b')
        pickle.dump(data_all,f)
        f.close()
        for keys,values in data_all.items():
            if 'out' not in keys:
                data_all[keys]=[]
        print('patrick',n,'outside temp',float(weather[0]))

    #I think the 'data' variable is a string... it certainly not an array so it'll need to be parsed / transformed before putting it in csv
    #below is example code of how to append to csv file. 
    #with open('document.csv','a') as fd:
        #fd.write(myCsvRow)


#Every time a disconnection happens he below event will happen
    #Use to create a log of all connection
    #Also maybe adjust the math for how a calculation is made. 
@sio.event
def disconnect(sid):
    print('disconnect ', sid)


#This tells the server to listen on port 5000
if __name__ == '__main__':

    f = open('raspi.txt', 'r')

    for l in f.readlines():
        dat = l.split(';')
        #print(dat[0])
        data_all[dat[0]] = []
    f.close()
    print(data_all)
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)


#pseudocode for thermostat control

#get count of active connections to server called 'cur_connects'
#get temperature readings from last cur_connect rows from csv file 
#compute average of those readings
# get time of day 
# get day of week
# use time of day and day of week to figure out what themo_setting should be
# if average is below therm_setting then GPIO(general-purpose input/output) pin 5 = HIGH else LOW
# when average = therm_setting+ .5 degree GPIO(general-purpose input/output) pin 5 = LOW
