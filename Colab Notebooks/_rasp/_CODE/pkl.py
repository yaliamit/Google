
import datetime
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as py
from matplotlib.ticker import MultipleLocator
import pickle
import os
import sys
from gas_process import correct_gas_readings
from scipy.interpolate import interp1d
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
    data_all['out_wind']=[]
    data_all['out_temp']=[]
    f.close()

    f=open(predir+'calibrate.txt','r')
    for l in f.readlines():
        dat=l.split(';')
        calibrate[dat[0]]=float(dat[1])

    print(calibrate)



def shorten_time(input_time):
    temp=':'.join(input_time.split(':')[0:2])
    return(temp)

def load_patrick(pname):
    file_name=predir+pname
    f=open(file_name,'rb')
    for keys,vals in data_all.items():
      data_all[keys]=[]
      new_data_all[keys]=None
    aa=[]
    i=0
    resolution={'minute':60,'hour':3600}
    while True:
      try:
        a=pickle.load(f)
      except:
        break

      for keys,vals in a.items():

        if keys=='out_temp' or keys=='out_wind':
            data_all[keys]+=list(np.array(vals).reshape(-1,2))
        else:
            data_all[keys]+=vals
    f.close()


    last_time=None
    first_time=None
    for keys,vals in data_all.items():
        if len(vals)>0:
            if len(vals)==1:
                vals=vals[0]
            ll=datetime.datetime.strptime(vals[-1][0],time_format)
            ff=datetime.datetime.strptime(vals[0][0],time_format)
            if last_time is None or ll>last_time:
                last_time=ll
            if first_time is None or ff<first_time:
                first_time=ff

    print(first_time,last_time)
    dt=last_time-first_time
    time_range=(dt.days*3600*24+dt.seconds)//resolution['minute']
    old_curr_hour=-1
    time_scale=np.array(range(time_range))
    len_time_scale=len(time_scale)
    D['dates']=[shorten_time(str(first_time))]
    D['dates_x']=[0]
    for keys,vals in data_all.items():
      if len(vals)>0:
          if len(vals) == 1:
              vals = vals[0]
          oct=0
          #first_time=datetime.datetime.strptime(vals[0][0],time_format)
          print(keys,len(vals))
          new_data_all[keys]=-np.ones((len_time_scale,2))
          for vv in vals:
            try:
                curr_time=datetime.datetime.strptime(vv[0],time_format)
            except:
                curr_time=datetime.datetime.strptime(vv[0],time_format_bakup)


            #print(curr_time,first_time)
            ct=(curr_time-first_time).seconds//resolution['minute']
            if ct>=len_time_scale:
                break
            if new_data_all[keys][ct,0]>=0:
                oct=ct
                continue
            else:
                new_data_all[keys][ct, 0] = ct
            new_data_all[keys][ct,1]=vv[1]
            if (ct-oct>1 and ((ct - oct<10) or keys=='out_temp' or keys=='out_wind')  and oct>0):
                sv=new_data_all[keys][oct,1]
                ev=float(vv[1])
                for i in range(oct,ct):
                    new_data_all[keys][i, 0] = i
                    new_data_all[keys][i, 1] = sv*float(ct-i)/(ct-oct)+ev*float(i-oct)/(ct-oct)
            oct=ct
            curr_hour=curr_time.hour
            if curr_hour != old_curr_hour and np.mod(curr_hour,6)==0 and keys=='1S-LIV':
                D['dates']+=[shorten_time(vv[0])]
                D['dates_x']+=[ct]
                old_curr_hour=curr_hour

      print(keys,'Done new_data')



def read_from_final(s,calibrate=None):
  f=open(predir+s,'rb')
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


def plot_stuff(gas):

    py.rcParams["figure.figsize"] = [14.50, 8.0]
    py.rcParams["figure.autolayout"] = True
    fig, AX= py.subplots(2,2,sharex=True,gridspec_kw={'height_ratios': [2,1]})
    fig.tight_layout()
    st=0
    ax11=AX[0,0];ax12=AX[0,1];ax21=AX[1,0];ax22=AX[1,1]

    ax1t = ax11.twinx()
    ax2t = ax12.twinx()
    ax3t=ax21.twinx()
    ax4t=ax22.twinx()
    ymin=100
    ymax=0
    ymin_out=100
    xmax=0
    xmin=100000




    colors = {'1S': 'blue', '2S': 'red', '3N':'green','2N':'orange','3S':'cyan', '1N':'purple','out':'black'}
    line_types={'LIV':'solid','DIN':'--','wind':'solid','temp':'solid'}

    for keys, vals in new_data_all.items():
        if vals is None:
            continue
        if keys == 'BACKUP-1' or keys=='2N-DIN-2':
            continue
        col=None
        if '-' in keys:
            key_col=colors[keys.split('-')[0]]
            key_lt=line_types[keys.split('-')[1]]
        else:
            key_col = colors[keys.split('_')[0]]
            key_lt = line_types[keys.split('_')[1]]
        tvals=-np.ones((vals.shape[0]+2,2))
        tvals[1:vals.shape[0]+1,:]=vals
        pos=(tvals[:,0]>0)*1
        dpos=np.diff(pos)
        wp=np.where(dpos==1)[0]
        we=np.where(dpos==-1)[0]
        ii=range(len(wp))
        for i,s,e in zip(ii,wp,we):
            if e-s<3:
                continue
            yy=np.minimum(vals[s:e-1,1],80)
            ysmoothed = gaussian_filter1d(yy, sigma=2)

            x=new_data_all[keys][s:e-1,0]
            xmax=np.maximum(xmax,x[-1])
            xmin=np.minimum(xmin,x[0])
            if keys=='out_temp':
                ax1t.plot(x,ysmoothed,label=keys,linewidth=.5,color='black')
                ax2t.plot(x, ysmoothed, label=keys, linewidth=.5, color='black')
                ax3t.plot(x, ysmoothed, label=keys, linewidth=.5, color='black')
                ax4t.plot(x, ysmoothed, label=keys, linewidth=.5, color='black')
                ymin_out=np.minimum(ymin_out,np.min(ysmoothed))
            elif keys=='out_wind':
                ygas=np.zeros(len(vals))
                xg = list(range(len(gas)))
                y = gas - gas[0]
                ysmoothed = gaussian_filter1d(y, sigma=4)
                f2 = interp1d(xg, ysmoothed, kind='linear')
                xnew = np.linspace(0, len(xg) - 1, num=len(ygas)-360)
                yygas = f2(xnew)
                ygas[360:]=yygas
                #y = days[i] - days[i][0]
                #ysmoothed = gaussian_filter1d(y, sigma=4)
                #ax21.plot(x,ysmoothed,label=keys,linewidth=1,color='red')
                #ax22.plot(x,ysmoothed,label=keys,linewidth=1,color='red')
            else:

                ysmoothed += calibrate[keys]
                ysmoothed=eliminate_peaks(ysmoothed)
                if 'S' in keys:
                    if i==0:
                        ax11.plot(x, ysmoothed,label=keys,color=key_col,linestyle=key_lt)
                    else:
                        ax11.plot(x, ysmoothed, color=key_col, linestyle=key_lt)
                else:
                    if i==0:
                        ax12.plot(x, ysmoothed, label=keys, color=key_col, linestyle=key_lt)
                    else:
                        ax12.plot(x, ysmoothed,  color=key_col, linestyle=key_lt)


                ymin = np.minimum(ymin, np.min(ysmoothed))
                ymax = np.maximum(ymax, np.max(ysmoothed))


    ax21.plot(ygas,label='gas',linewidth=1,color='red')
    ax21.set_xlim(left=0, right=len(gas))
    ax22.plot(ygas,label='gas',linewidth=1,color='red')

    ax11.set_xlim(left=0,right=xmax)
    ax11.set_ylim(bottom=ymin-1,top=ymax)
    ax12.set_xlim(left=0,right=xmax)
    ax12.set_ylim(bottom=ymin-1,top=ymax)


    ax1t.set_ylim(bottom=ymin_out)
    ax2t.set_ylim(bottom=ymin_out)
    dates_xx=np.array(D['dates_x'])-st
    text=[]
    py.rcParams['font.size'] = 10
    text=[]
    for xx,d in zip(dates_xx,D['dates']):
      #(xx)
      d='-'.join(d.split('-')[1:])
      text+=[d] #ax.text(xx,ymin-2.8,d,fontsize=6,rotation=-45,rotation_mode='anchor')]

    ax11.xaxis.set_major_locator(MultipleLocator(120))
    #ax.set_xticklabels(['','','','','',''])
    ax21.tick_params(which='major', axis='x',length=12,labelsize=10)
    ax22.tick_params(which='major', axis='x',length=12,labelsize=10)

    ax11.xaxis.set_minor_locator(MultipleLocator(60))
    ax11.tick_params(which='minor', length=5)
    fig.subplots_adjust(bottom=0.3,right=.65)
    py.subplots_adjust(left=0.1,
                       bottom=0.3,
                       right=0.65,
                       top=0.9,
                       wspace=0.4,
                       hspace=0.4)
    xt = ax11.get_xticks()
    xta = []
    for xtb in xt:
        if xtb >= xmin and xtb <= xmax:
            xta += [xtb]
    xta = np.int32(np.append(np.array(xta), dates_xx))
    xtb=xta//60
    xtl = xtb.tolist()
    xtl[-len(dates_xx):] = text #list(str(' ') * len(dates_xx))

    ax11.set_xticks(xta)
    ax21.set_xticklabels(xtl,ha='left',rotation=-45)
    ax22.set_xticklabels(xtl,ha='left',rotation=-45)

    py.title('Temps')
    #py.xlabel('Minutes')
    ax11.set_ylabel('Indoor Temp')
    ax21.set_xlabel('Hours',labelpad=0)
    ax21.set_ylabel('Gas consumption')
    ax2t.set_ylabel('Outdoor Temp')
    ax4t.set_ylabel('Outdoor Temp')
    ax11.legend(loc=(1.05, .7))
    ax12.legend(loc=(1.05, .7))
    ax2t.legend(loc=(1.05,0))
    ax21.legend(loc=(1.1,0))
    DD='_'.join(D['dates'][0].split(' '))
    DD='_'.join(DD.split(':'))
    fig_name=predir+'_BAK/final_dump_'+DD+".png"
    py.savefig(fig_name)
    fig_name=predir+'today_dump.png'
    py.savefig(fig_name)
    f = open(predir + '_BAK/final_'+DD+'.pkl', 'wb')
    pickle.dump({'final_data_all':new_data_all,'D':D}, f)
    f.close()
    #f = open(predir + 'patricks_dump_'+DD+ ".pkl", 'wb')
    #pickle.dump(data_all, f)
    #f.close()


if len(sys.argv)==3:
    load_rasp_list(file_name=sys.argv[1])
    new_data_all, D= read_from_final(sys.argv[2])
    plot_stuff(old=False)
else:
    pname='patricks_dump.pkl'
    if len(sys.argv)==2:
        pname=sys.argv[1]
    load_rasp_list()
    load_patrick(pname)
    days,dates=correct_gas_readings(predir)

    plot_stuff(days[-1])
#plot_stuff()
