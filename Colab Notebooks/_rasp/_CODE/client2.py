import socketio
import os
import glob
import time
import datetime as dt
import board
import adafruit_dht
import sys
import os


dhtDevice = adafruit_dht.DHT22(board.D4,use_pulseio=False)

 
#os.system('modprobe w1-gpio')
#os.system('modprobe w1-therm')
 
#base_dir = '/sys/bus/w1/devices/'
#device_folder = glob.glob(base_dir + '28*')[0]
#device_file = device_folder + '/w1_slave'
 
def read_temp_raw():
   try:
      temp_c=dhtDevice.temperature
      return temp_c 
   except RuntimeError as error:
      #print(error.args[0])   
      time.sleep(2.0)
      return None
   except Exception as error:
      print('Failed to read thermometer')
      dhtDevice.exit()
      raise error

def read_temp():
    temp_c  = read_temp_raw()
    temp_f=None
    if temp_c is not None:
         temp_f = temp_c * 9.0 / 5.0 + 32.0
    return temp_f
	
#while True:
#	print(read_temp())	
#	time.sleep(1)

sio = socketio.Client()

def sensor_reading():
    room=os.uname()[1]
    print('room',room)
    while True:
        reading = read_temp()
        if reading is not None:
            ct = str(dt.datetime.now())
            sio.emit('my_message', {'temp': reading, 'loc': room, 'ts':ct})#'ts':ct
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

#attempts=0
#while attempts<10:
#    try:
sio.connect('http://192.168.0.200:5000')
#except:
#        print('Failed to connect to server at attempt',attempts)
#        attempts+=1
#        continue
#print('Exiting after 10 attemps')
#exit()
