import os
from os import walk
import numpy as np

def make_tex(dir='save/'):
    if 'Linux' in os.uname():
        predir = '/ME/My Drive/'
    else:
        predir = '/Users/amit/Google Drive/'
    datadirs = predir + 'Colab Notebooks/FA/'
    fl = []
    dira=datadirs+dir
    for (dirpath, dirnames, filenames) in walk(dira):

        for ff in filenames:
            if 'jpg' in ff and 'acc' in ff:
                fl.append(ff)
        break
    fl.sort()
    f=open(datadirs+dir+'Res.tex','w')
    fh=open(datadirs+'head.tex')

    for l in fh:
        f.write(l)

    for i,ff in enumerate(fl):
        f.write("\\includegraphics[width=2.5in]{"+ff+"}\n")

        if np.mod(i,2)==1:
            f.write("\n")

    f.write('\\end{document}')
    f.close()

    #com = "cat "+datadirs+"head.tex " + datadirs+dir +"temp.tex > " + datadirs+dir + "Res.tex"
    #print(com)
    #os.system(com)
