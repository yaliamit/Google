import os
from os import walk
import numpy as np

def make_tex(dir):

    fl = []
    for (dirpath, dirnames, filenames) in walk(dir):

        for ff in filenames:
            if 'jpg' in ff:
                fl.append(ff)
        break

    f=open(dir+'temp.tex','w')

    for i,ff in enumerate(fl):
        f.write("\\includegraphics[width=2.5in]{"+ff+"}\n")

        if np.mod(i,2)==1:
            f.write("\n")

    f.write('\\end{document}')
    f.close()

    com = "cat head.tex " + dir +"temp.tex > " + dir + "Res.tex"
    os.system(com)
