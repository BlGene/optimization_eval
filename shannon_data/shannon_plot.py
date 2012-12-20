import numpy
import pylab

f_name = "optimization_sw.csv"

ax_list = [ pylab.subplot(4,1,1),
    pylab.subplot(4,1,2),
    pylab.subplot(4,1,3),
    pylab.subplot(4,1,4)]


a = numpy.loadtxt(f_name,delimiter=',')

ax_label = ["Best","Std. Dev", "N<-1","Distance"]

for j in range(8):
    for i in range(4):
        ax_list[i].plot(a[:,j*4+i+1])
        #ax_list[i].grid(True)
        

for i in range(4):
    ax_list[i].set_ylabel(ax_label[i])
    #ax_list[i].set_xscale("log")
    
    if i < 3:
        ax_list[i].xaxis.set_visible(False)
        
pylab.show()

