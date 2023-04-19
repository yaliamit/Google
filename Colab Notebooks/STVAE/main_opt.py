import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from aux_colab import seq, run_net, save_net
from data import get_pre
from layers import *

predir=get_pre()
import os





# if 'Linux' in os.uname():
#     from google.colab import drive
#     drive.mount('/ME')
#     predir='/ME/My Drive/'
# else:
#     predir='/Users/amit/Google Drive/'


datadirs=os.path.join(predir,'Colab Notebooks/STVAE/')
sys.path.insert(1, datadirs)
sys.path.insert(1, datadirs+'_CODE')

print(sys.argv)
if not torch.cuda.is_available():
    device=torch.device("cpu")
else:
    if len(sys.argv)==1:
        s="cuda:"+"0"
    else:
        s="cuda:"+sys.argv[1]
    device=torch.device(s)
print(device)


count_non=0
with open('junk','w') as f:
  for a in sys.argv:
    if '--' in a:
        pass
    else:
        count_non+=1


if count_non<3:
    par_file='t_par'
else:
    par_file=sys.argv[2]
    print(par_file)

aa=[]
with open(par_file+'.txt','r') as f:
    aap=f.readlines()
    for ap in aap:
        if '#' not in ap:
            aa.append(ap)

with open(par_file+'_temp.txt','w') as f:
    for a in aa:
        f.write(a)
        print(a)
    f.write('\n')
    for a in sys.argv:
        if '--' in a:
            f.write(a+'\n')
            print(a+'\n')




temp_file=par_file+'_temp'
if count_non<4:
    net,_,args=run_net(temp_file, device)
    #if not args.run_existing:
    #    save_net(net,temp_file,predir)
else:
    tlay=None
    toldn=None
    if len(sys.argv)>4:
        tlay=sys.argv[4]
        toldn=sys.argv[5]
    seq(temp_file,predir, device, tlay=tlay, toldn=toldn)



print("hello")


