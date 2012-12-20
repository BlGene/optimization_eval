import os
import numpy
import pylab

def makePlot(choice,lines,plot_names):
    
    data_dir = 'data_' + choice 
    
    f_list = os.listdir(data_dir)
    inspyred_list = []
    pygmo_list = []
    
    for f in f_list:
        if f.startswith("inspyred"):
            inspyred_list.append(f)
        elif f.startswith("PyGMO"):
            pygmo_list.append(f)
    
    
    #f_list = inspyred_list
    #f_list = pygmo_list
           
    ax_list = [ pylab.subplot(5,1,1),
        pylab.subplot(5,1,2),
        pylab.subplot(5,1,3),
        pylab.subplot(5,1,4)]
    
    ax_label = ["Best","Std. Dev", "N<-1","Distance"]
    
    
    for name in f_list:
        use = False
        for i in lines:
            if name.startswith(i):
                use = True
        if not use:
            continue
        
        a = numpy.loadtxt(os.path.join(data_dir,name))
        
    
        newname = " ".join(name.split("_")[:-1]).lower()
        print  newname
        
        for i in range(4):
            print i
            ax_list[i].plot(a[:,i],label=newname)
            #ax_list[i].grid(True)
            ax_list[i].set_ylabel(ax_label[i])
            if i < 3:
                ax_list[i].xaxis.set_visible(False)
    
    #-------------------
    
    if choice == "hard":
        f_name = os.path.join("shannon_data","optimization_sw.csv")
    elif choice == "easy":
        f_name = os.path.join("shannon_data","optimization_sw_w0-25.csv")
    else:
        raise RuntimeError
    
    b = numpy.loadtxt(f_name,delimiter=',')
    
    names = ["Matlab_Pattern_Search", "Matlab_GA","Matlab_SA","Matlab_PSO",
             "Matlab_LUS","Matlab_DE", "Matlab_CMA_ES", "Matlab_MCS"]
    
    for j in range(8):
        use = False
        for i in lines:
            if names[j].startswith(i):
                use = True
        if not use:
            continue
        
        for i in range(4):
            ax_list[i].plot(b[:,j*4+i+1],label=names[j])
            
            ax_list[i].set_xscale("log")
            if i < 3:
                ax_list[i].xaxis.set_visible(False)
    
    
    from matplotlib.font_manager import FontProperties
    
    fontP = FontProperties()
    fontP.set_size('small')
    
    pylab.legend(bbox_to_anchor=(0, -1.1, 1, 1),loc=3,mode="expand", ncol=4, borderaxespad=0.,prop=fontP,
                 fancybox=True, shadow=True)
    
    #pylab.draw()
    #pylab.show()
    pylab.savefig(plot_name+"_"+choice+".png")
    pylab.clf()
    
if __name__ == "__main__":
    choice = 'easy'
    
    lines = ['PyGMO_DE','PyGMO_JDE','PyGMO_MDE_PBX','inspyred_DEA','inspyred_EDA','inspyred_EA','Matlab_GA',]#'Matlab_DE']
    plot_name = "EA"
    makePlot(choice,lines,plot_name)

    lines = ['PyGMO_PSO','PyGMO_BEE_COLONY','inspyred_PSO','Matlab_PSO']
    plot_name = "Swarm"
    makePlot(choice,lines,plot_name)

    lines = ['PyGMO_IHS','Matlab_SA','Matlab_Pattern_Search','Matlab_LUS','Matlab_MCS','Matlab_CMA_ES']
    plot_name = "Misc"
    makePlot(choice,lines,plot_name)

    lines = ['PyGMO_IHS','Matlab_MCS','PyGMO_PSO_GEN','inspyred_DEA','inspyred_EDA']
    plot_name = "Best"
    makePlot(choice,lines,plot_name)

    #---------------------------------
    choice = 'hard'

    lines = ['PyGMO_DE','PyGMO_JDE','PyGMO_MDE_PBX','inspyred_DEA','inspyred_EDA','inspyred_EA','Matlab_GA',]#'Matlab_DE']
    plot_name = "EA"
    makePlot(choice,lines,plot_name)

    lines = ['PyGMO_PSO','PyGMO_BEE_COLONY','inspyred_PSO','Matlab_PSO']
    plot_name = "Swarm"
    makePlot(choice,lines,plot_name)

    lines = ['PyGMO_IHS','Matlab_SA','Matlab_Pattern_Search','Matlab_LUS','Matlab_MCS','Matlab_CMA_ES']
    plot_name = "Misc"
    makePlot(choice,lines,plot_name)

    lines = ['PyGMO_IHS','PyGMO_PSO','inspyred_DEA']
    plot_name = "Best"
    makePlot(choice,lines,plot_name)

   
    

    
