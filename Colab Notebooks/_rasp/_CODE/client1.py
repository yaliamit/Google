import socketio
import sys
import os
import glob
import time
import datetime as dt


os.system('modprobe w1-gpio')
os.system('modprobe w1-therm')
 
base_dir = '/sys/bus/w1/devices/'
device_folder = glob.glob(base_dir + '28*')[0]
device_file = device_folder + '/w1_slave'
 
def read_temp_raw():
    f = open(device_file, 'r')
    lines = f.readlines()
    f.close()
    return lines
 
def read_temp():
    lines = read_temp_raw()
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        lines = read_temp_raw()
    equals_pos = lines[1].find('t=')
    if equals_pos != -1:
        temp_string = lines[1][equals_pos+2:]
        temp_c = float(temp_string) / 1000.0
        temp_f = temp_c * 9.0 / 5.0 + 32.0
        return temp_f
	
#while True:
#	print(read_temp())	
#	time.sleep(1)

sio = socketio.Client()

def sensor_reading():
    room=os.uname()[1]
    while True:
        reading = read_temp()
        ct = str(dt.datetime.now())
        message={'temp': reading, 'loc': room, 'ts':ct}
        sio.emit('my_message', message)
        sio.sleep(15)

@sio.event
def connect():
    print('connection established')
    sio.start_background_task(sensor_reading)

def connect_error():
    print("The connection failed!")

@sio.event
def disconnect():
    print('disconnected from server and rebooted')
    os.system('sudo reboot')

sio.connect('http://192.168.0.200:5000')
