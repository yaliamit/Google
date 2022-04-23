import numpy as np
import pylab as py
from scipy.ndimage.filters import gaussian_filter1d
import os

def correct_gas_readings(predir):
    f = open(predir + 'gas_readings.txt')
    fn = open(predir + 'corrected_gas_readings.txt', 'w')
    reads=[]
    days=[]
    dates=[]
    thisday=[]
    old_string_num=''
    start=True
    lastval=None
    for l in f.readlines():

        dat=l.split(':')
        string_num=''.join((dat[1].split('\n')[0].split(' ')))
        if not start:
            for j in range(5):
                if old_string_num[j] == '8' and string_num[j] == '0':
                    string_num = string_num[:j] + '8' + string_num[j + 1:]
            for j in range(4):
                if old_string_num[j] == '5' and string_num[j] == '6' and int(old_string_num[j+1])<9:
                    string_num = string_num[:j] + '5' + string_num[j + 1:]
            # Same last digit, assume nothing has changed.
            if old_string_num[-1]==string_num[-1]:
                read=[dat[0],lastval]
            else:

                # for j in range(5):
                #     if old_string_num[j]=='8' and string_num[j]=='0':
                #        string_num[j]==8
                num=np.int64(string_num)
                # You record a unit increment in the reading, that must be correct.
                if num-lastval==1:
                    read=[dat[0],num]
                # You record a decrease in only the last digit, everything else the same assume nothing has changed.
                #elif string_num[0:4]==old_string_num[0:4] and int(string_num[4])<int(old_string_num[4]):
                elif string_num[3]==old_string_num[3] and int(string_num[4])<int(old_string_num[4]):
                    read=[dat[0],lastval]
                elif string_num[2]==old_string_num[2] and int(string_num[3:])<int(old_string_num[3:]):
                    read=[dat[0],lastval]

                # An increment of greater or equal to 1 in the last digit (mod 10), assume increment of 1.
                elif int(string_num[3:])-int(old_string_num[3:])>=1 or (int(old_string_num[4]=='9') and int(string_num[4]=='0')):
                    read=[dat[0],lastval+1]
                # Increment of 1 in last two digits assume increment of 1.
                elif int(string_num[3:5])-int(old_string_num[3:5])==1:
                    read = [dat[0], lastval + 1]
                elif int(string_num[2:5])-int(old_string_num[2:5])==1:
                    read = [dat[0], lastval + 1]
                elif int(string_num[1:5])-int(old_string_num[1:5])==1:
                    read = [dat[0], lastval + 1]
                elif string_num[1]==old_string_num[1] and int(string_num[2:])<int(old_string_num[2:]):
                    read=[dat[0],lastval]
                else:
                    print(dat[0],old_string_num,string_num,'IDK')
                print(dat[1],read[1])
        else:
            read=[dat[0],np.int64(string_num)]
            start=False
        thisday+=[read[1]]
        lastval=read[1]
        old_string_num=str(read[1])
        ss=' '.join(list(old_string_num))
        fn.write(dat[0]+": "+ss+'\n')
        if dat[0][-5:]=='22-55':
            dates+=[dat[0][:-5]]
        if dat[0][-5:]=='05-00':
            days+=[thisday]
            thisday=[]
    days+=[thisday]
    dates+=[dat[0][:-5]]

    days=days[1:]
    dates=dates[1:]

    f.close()
    fn.close()
    return days, dates

if __name__ == '__main__':
    #os.system('scp sol@192.168.0.200:gas_readings.txt ../.')
    predir = '../'

    days,dates=correct_gas_readings(predir)

    ndays=len(dates)
    numcols=3
    nrows=ndays//numcols
    if ndays % numcols != 0:
        nrows+=1

    py.rcParams["figure.figsize"] = [8.,2*nrows]
    fig, axs=py.subplots(nrows,numcols,sharey=True)
    ftemp=open(predir+'M_temps.txt')
    temps=ftemp.readlines()

    for i in range(ndays):
            rn=i//numcols
            cn=i%numcols
            y=days[i]-days[i][0]
            y=gaussian_filter1d(y, sigma=4)
            axs[rn][cn].plot(y)
            tt=temps[i].strip('\n')
            axs[rn][cn].set_xlabel('-'.join(dates[i].split('-')[1:3])+' '+tt.split(':')[1]+'F')
    fig.tight_layout()
    #fig.show()
    py.savefig("gas_plots")
    print("hello")
