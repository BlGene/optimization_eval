import os.path
import time
from math import pi,exp
from random import gauss,Random,uniform

from PyGMO.problem import base
from PyGMO import algorithm,island

import inspyred

create_count = 0
exec_count = 0
eval_lock = False

sigma_w = .1
sigma_a = .5 * sigma_w
sigma_b = .2
sigma_d = .1

def gaussNoise(x_vect,optimum):


    f = 1;
    for i in range(len(x_vect)):
        f = f * exp(-0.5*( (x_vect[i]-optimum[i]-sigma_a*gauss(0,1))/sigma_w )**2)
    
    f = -f*(1+sigma_b*gauss(0,1))+sigma_d*gauss(0,1)
    
    return f
        

class PyGMO_GaussNoise_Problem(base):
    """
    N-dimensional Gauss with Noise, implemented purely in Python.
    
    USAGE: my_problem(dim = 10)
    
    * dim problem dimension
    """
    def __init__(self, dim = 10,lb=-1,ub=1):
        #global create_count
        #create_count += 1
        
        #print 'i>!'
        
        #First we call the constructor of the base class telling
        #essentially to PyGMO what kind of problem to expect (1 objective, 0 contraints etc.)
        super(PyGMO_GaussNoise_Problem,self).__init__(dim)
        
        #then we set the problem bounds (in this case equal for all components)
        self.set_bounds(lb,ub)
        
        #and we define some additional 'private' data members (not really necessary in
        #this case, but ... hey this is a tutorial)
        self.__dim = dim
        self.__eval_count = 0
        self.__optimum = [uniform(lb+sigma_w,ub-sigma_w) for _ in range(dim)]
        
    #We reimplement the virtual method that defines the objective function.
    def _objfun_impl(self,x):

        self.__eval_count +=1
        #print 'e>!',self.__eval_count

        f = gaussNoise(x,self.__optimum)
        
        #note that we return a tuple with one element only. In PyGMO the objective functions
        #return tuples so that multi-objective optimization is also possible.
        return (f,)

    #Finally we also reimplement a virtual method that adds some output to the __repr__ method
    def human_readable_extra(self):
        return "\n\t Problem dimension: " + str(self.__dim)

    def get_evals(self):
        return "Evals: " +str(self.__eval_count)

    def get_optimum(self):
        return self.__optimum


class Inspyred_GaussNoise_Problem(inspyred.benchmarks.Benchmark):
    def __init__(self, dim=3,lb=-1,ub=1):
        
        self.dim = dim
        self.my_lb = lb 
        self.my_ub = ub
        
        inspyred.benchmarks.Benchmark.__init__(self, dim)
        self.bounder = inspyred.ec.Bounder(lb,ub)
        
        self.maximize = False
        
        self.global_optimum = [uniform(lb+sigma_w,ub-sigma_w) for _ in range(dim)]

        self.record  = False
        self.record_list = []

        self.all_x = []
        self.all_f = []
        
    def generator(self, random, args):
        return [random.uniform(self.my_lb,self.my_ub) for i in range(self.dim)]

    def evaluator(self,candidates,args):
        fitness = []
        
        for cs in candidates:
            f = gaussNoise(cs,self.global_optimum)
            fitness.append(f)
                
        self.all_x.extend(candidates)
        self.all_f.extend(fitness)
        
        return fitness


def run_eval(lib_name,algos):
    import pylab
    import numpy
    
    for name,algo in algos:
        
        collect_all_x = []
        collect_all_f = []
        collect_best_f_so_far = []
        collect_all_dx = []
        
        print name,"\t\tRun:",
                
        for i in range(repeats):
            if (i%10) == 0:
                print i,
            
            if lib_name == "PyGMO":
                #---------------------------------------------------
                prob = PyGMO_GaussNoise_Problem(dim,lb,ub)
                isl = island(algo,prob,pop_size)
                pop = isl.population
                
                best = [pop.champion.f]
                best_x = [pop.champion.x]
                all_x = [p.cur_x for p in pop]
                all_f = [p.cur_f for p in pop]
                
                for _ in range(number_of_generations):
                    pop = algo.evolve(pop)
            
                    best.append(pop.champion.f)
                    best_x.append(pop.champion.x)
                    all_x.extend([p.cur_x for p in pop])
                    all_f.extend([p.cur_f for p in pop])
            
                    #print pop.champion.f
                optimum = prob.get_optimum()
                    
                #-----------------------------------------------------
            elif lib_name == "inspyred":
                #-----------------------------------------------------
                ea = algo
                ea.terminator = inspyred.ec.terminators.evaluation_termination
                
                problem = Inspyred_GaussNoise_Problem(dim,lb,ub)
    
                final_pop = ea.evolve(generator=problem.generator, 
                              evaluator=problem.evaluator, 
                              pop_size=30, 
                              bounder=problem.bounder,
                              maximize=problem.maximize,
                              max_evaluations=total_evaluations)
                
                all_x = problem.all_x
                all_f = problem.all_f
                optimum = problem.global_optimum
                #-----------------------------------------------------
            else:
                raise RuntimeError("Unknown lib_name")
                
            all_dx = numpy.sum((numpy.array(all_x) - optimum)**2,axis=1)
            
            best_f_so_far = [all_f[0]]
            for f in all_f[1:]:
                best_f_so_far.append(min(f,best_f_so_far[-1]))
            '''
            if(i == 0):
                pylab.plot(best_f_so_far)
                pylab.plot(all_f)
                pylab.plot(all_dx)
                pylab.show()
            
            '''
            collect_all_x.append(all_x)
            collect_all_f.append(all_f)
            collect_all_dx.append( all_dx)
            collect_best_f_so_far.append(best_f_so_far)
            
        print "." #end repeats
        
        #get into numpy form, remove last dimension that comes from
        #multiobjective optimization compatibility
        collect_all_f = numpy.squeeze(numpy.array(collect_all_f))
        collect_best_f_so_far = numpy.squeeze(numpy.array(collect_best_f_so_far))
        collect_all_dx = numpy.array(collect_all_dx)
        
        #do calculations, all on best_f_so_far other than distance
        median_bf = numpy.median(collect_best_f_so_far,axis=0)
        stddev_bf = numpy.std(collect_best_f_so_far,axis=0)
        under_minusone_bf = numpy.sum( collect_best_f_so_far < -1 , axis=0)/50.0
        
        distance =  numpy.median(collect_all_dx,axis=0)
        
        file_array = numpy.c_[median_bf,stddev_bf,under_minusone_bf,distance]
        
        new_name = name+"_"+str(repeats)+"reps.txt"
        fname = os.path.join(data_dir,new_name)
        
        numpy.savetxt(fname,file_array)
        
    print create_count,exec_count
    
def run_pygmo():
    algos = [('PyGMO_DE',algorithm.de(gen=1)),
        ('PyGMO_DE_v1',algorithm.de(gen=1,f=.2,cr=1.0,variant=4)),
        #('PyGMO_DE_v2',algorithm.de(gen=1,f=.1,cr=1.0,variant=4)),
        #('PyGMO_DE_v3',algorithm.de(gen=1,f=.3,cr=1.0,variant=4)),
        #('PyGMO_DE_v4',algorithm.de(gen=1,f=.2,cr=0.9,variant=4)),
        ('PyGMO_JDE',algorithm.jde(gen=1)),
        ('PyGMO_MDE_PBX',algorithm.mde_pbx(gen=1)),
        ('PyGMO_PSO',algorithm.pso(gen=1,variant=5)),
        ('PyGMO_PSO_GEN',algorithm.pso_gen(gen=1)),
        ('PyGMO_BEE_COLONY',algorithm.bee_colony(gen=1)),        
        ('PyGMO_IHS',algorithm.ihs(iter=pop_size))]
        
    run_eval("PyGMO",algos)
    
def run_inspyred(prng=None):
    if prng is None:
        prng = Random()
        prng.seed(time.time()) 
        
    #Too bad
    #ea = inspyred.ec.ES(prng)
    #ea = inspyred.ec.SA(prng)

    algos = [ ('inspyred_GA',inspyred.ec.GA(prng)),
              ('inspyred_DEA',inspyred.ec.DEA(prng)),
              ('inspyred_EDA',inspyred.ec.EDA(prng)),
              ('inspyred_PSO',inspyred.swarm.PSO(prng))]
              
    run_eval("inspyred",algos)
              

if __name__ == "__main__":

    if sigma_w <.2:
        data_dir = "data_hard"
    else:
        data_dir = "data_easy"
    
    repeats = 50
    
    dim = 3
    pop_size = 30
    total_evaluations = 900
    number_of_generations= int(total_evaluations/pop_size) -1

    lb = 0
    ub = 1
    
    print "Running with w = ",sigma_w
    print "Dir:",data_dir
    
    run_inspyred()
    run_pygmo()
    
    
    
