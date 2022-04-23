
import datetime
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as py
from matplotlib.ticker import MultipleLocator
import pickle
import os
import sys
from sklearn.linear_model import LinearRegression

print(sys.platform)

#predir='/Users/amit/Google Drive/Colab Notebooks/_rasp/'
if sys.platform == 'linux':
    predir=''
else:
    predir='../'
time_format="%Y-%m-%d %H:%M:%S.%f"
time_format_bakup="%Y-%m-%d %H:%M:%S"
time_format_short=("%Y-%m-%d %H:%M")
data_all={}
new_data_all={}
calibrate={}
D={'dates':[],'dates_x':[]}

def load_rasp_list(file_name='raspi.txt'):
    f = open(predir+file_name, 'r')
    for l in f.readlines():
            dat = l.split(';')
            data_all[dat[0]] = []
            new_data_all[dat[0]]=None
    f.close()

    f=open(predir+'calibrate.txt','r')
    for l in f.readlines():
        dat=l.split(';')
        calibrate[dat[0]]=float(dat[1])

    print(calibrate)



def shorten_time(input_time):
    temp=':'.join(input_time.split(':')[0:2])
    return(temp)


def read_from_final(s,calibrate=None):
  f=open(s,'rb')
  ss=pickle.load(f)
  newd=ss['final_data_all']
  return ss['final_data_all'],ss['D']

def eliminate_peaks(y):

    d1=np.diff(y)
    d2=np.diff(d1)

    ii=np.where(np.abs(d2)>.4)[0]
    if len(ii)==0:
        return y
    ll=[]
    lasti=ii[0]
    ymax=np.abs(d2[ii[0]])
    ymaxes=[]
    imax=lasti
    for j,i in enumerate(ii):
        if i-lasti<=3:
            lasti=i
            ymax=np.maximum(ymax,np.abs(d2[i]))
            imax=i
        else:
            ymaxes+=[np.array((ymax,imax))]
            ymax=np.abs(d2[i])
            lasti=i
            imax=lasti

    ymaxes += [np.array((ymax, imax))]

    for vv in ymaxes:
        j=vv[1]
        jstart=int(np.maximum(j-10,0))
        jend=int(np.minimum(j+10,len(y)-1))
        ystart=y[jstart]
        yend=y[jend]
        for j in range(jstart,jend):
            y[j]=(yend*(j-jstart)+ystart*(jend-j))/(jend-jstart)

    return y


def plot_stuff():
    Y={}
    for keys, vals in new_data_all.items():
        Y[keys]=0

    start_time=720
    for keys, vals in new_data_all.items():
        if vals is None:
            continue
        if keys == 'BACKUP-1' or keys=='2N-DIN-2':
            continue


        tvals=-np.ones((vals.shape[0]+2,2))
        tvals[1:vals.shape[0]+1,:]=vals
        pos=(tvals[:,0]>0)*1
        dpos=np.diff(pos)
        wp=np.where(dpos==1)[0]
        we=np.where(dpos==-1)[0]
        ii=range(len(wp))
        leng=0
        for i,s,e in zip(ii,wp,we):
            if e-s<3 or e<start_time:
                continue
            ss=np.maximum(start_time,s)
            yy=np.minimum(vals[ss:e-1,1],80)
            ysmoothed = gaussian_filter1d(yy, sigma=2)

            leng+=len(ysmoothed)
            if 'out' in keys:
                Y[keys]+=np.sum(ysmoothed)
            else:
                ysmoothed += calibrate[keys]
                ysmoothed = eliminate_peaks(ysmoothed)
                Y[keys]+=np.sum(ysmoothed-70)
        if leng>0:
            Y[keys]/=leng
        else:
            Y[keys]=None


    return(Y)



load_rasp_list()
aa=os.listdir(predir+'_FINALS')
aa.sort()
dates=[]
YY=None
for a in aa:

    if 'pkl' in a:
        new_data_all, D = read_from_final(predir+'_FINALS/'+a)
        if '1S-LIV' in new_data_all:
            continue
        print(a)
        Y=plot_stuff()
        if YY==None:
            YY={}
            for keys,vals in Y.items():
                YY[keys]=[]
                YY[keys]+=[vals]
        else:
            for keys,vals in Y.items():
                YY[keys]+=[vals]
        dd=a.split('_')[1]
        dates+=[dd]

py.rcParams["figure.figsize"] = [12.,2*6]
fig, axs=py.subplots(3,4,sharey=True)
temps=YY['out_temp']
i=0
for keys,vals in sorted(YY.items()):
    if 'out' not in keys and '2N-DIN-2' not in keys:
        print(keys)
        rn = i // 4
        cn = i % 4
        xy=[z for z in zip(temps,vals) if z[1] is not None]
        xy=np.array(xy)
        xx=xy[:,0].reshape(-1,1)
        reg = LinearRegression().fit(xx, xy[:,1])
        y_pred = reg.predict(xx)
        axs[rn][cn].scatter(xy[:,0],xy[:,1],color='black')
        axs[rn][cn].plot(xy[:,0], y_pred, color="blue", linewidth=3)
        axs[rn][cn].set_xlabel(keys)
        i+=1
fig.tight_layout()
py.savefig('scats')
print('Hello')
#plot_stuff()
