from PyGMO import problem, algorithm, island,__extensions__
from numpy import mean, std
from math import exp,pi
             
class deJong_problem_stochastic(problem.base_stochastic):
    """
    Noisy De Jong (sphere) function implemented purely in Python.
    
    USAGE: my_problem_stochastic(dim = 10, seed=0)
    
    * dim problem dimension
    * seed initial random seed
    """
    def __init__(self, dim = 10, seed = 0):
         #First we call the constructor of the base stochastic class. (Only
         #unconstrained single objective problems can be stochastic in PyGMO)
         super(my_problem_stochastic,self).__init__(dim, seed)
    
         #then we set the problem bounds (in this case equal for all components)
         self.set_bounds(-5.12,5.12)
    
         #and we define some additional 'private' data members (not really necessary in
         #this case, but ... hey this is a tutorial)
         self.__dim = dim
    
    def _objfun_impl(self,x):
         from random import random as drng
         from random import seed
    
         #We initialize the random number generator using the
         #data member seed (in base_stochastic). This will be changed by suitable
         #algorithms when a stochastic problem is used. The mod operation avoids overflows
    
         seed(self.seed)
    
         #We write the objfun using the same pseudorandonm sequence
         #as long as self.seed is unchanged.
         f = 0;
         for i in range(self.__dim):
                 noise = (2 * drng() - 1) / 10
                 f = f + (x[i] + noise)*(x[i] + noise)
         return (f,)
    def human_readable_extra(self):
         return "\n\tSeed: " + str(self.seed)
         
class gauss_problem_stochastic(problem.base_stochastic):
    """
    Noisy Gauss function implemented purely in Python.
    
    USAGE: my_problem_stochastic(dim = 10, seed=0)
    
    * dim problem dimension
    * seed initial random seed
    """
    def __init__(self, dim = 4, seed = 0):
        #First we call the constructor of the base stochastic class. (Only
        #unconstrained single objective problems can be stochastic in PyGMO)
        super(gauss_problem_stochastic,self).__init__(dim, seed)
        
        #then we set the problem bounds (in this case equal for all components)
        self.set_bounds(0,pi)
        
        #and we define some additional 'private' data members (not really necessary in
        #this case, but ... hey this is a tutorial)
        self.__dim = dim
        self.__eval_count = 0
        self.__generation = 1
        self.__best_eval_f = 10**10
        self.__compare = min
        self.__eval_history = []
    
    def _objfun_impl(self,x):
        from random import random as drng
        from random import gauss
        from random import seed
        
        #We initialize the random number generator using the
        #data member seed (in base_stochastic). This will be changed by suitable
        #algorithms when a stochastic problem is used. The mod operation avoids overflows
        
        seed(self.seed)
        
        #We write the objfun using the same pseudorandonm sequence
        #as long as self.seed is unchanged.
        w = .25
        a = 0.2
        b = 0.2
        f = 1;
        for i in range(self.__dim):
            f = f * exp(-0.5*( (x[i]-gauss(pi/2,a))/w )**2)
        f = -f*gauss(1,b)
        
        self.__eval_count += 1
        #print self.__eval_count,

        self.__best_eval_f = self.__compare(f,self.__best_eval_f)
        self.__eval_history.append((self.__eval_count,self.__best_eval_f,self.__generation))
        
        return (f,)
        
    def human_readable_extra(self):
        return "\n\tSeed: " + str(self.seed)
        #return 'Evals: '+str(self.__eval_count)

    def incr_gen(self):
        self.__generation += 1
        
       
    def get_evals(self):
        return self.__eval_count

    def get_history(self):
        return self.__eval_history

def run_test(n_trials=200, pop_size = 20, n_gen = 500):

    number_of_trials = n_trials
    number_of_individuals = pop_size
    number_of_generations = n_gen
    
    prob_list = [problem.schwefel(dim = 10),
        problem.michalewicz(dim = 10),
        problem.rastrigin(dim = 10),
        problem.rosenbrock(dim = 10),
        problem.ackley(dim = 10),
        problem.griewank(dim = 10)]

    if __extensions__['gtop']:
        prob_list.append(problem.cassini_1())
        prob_list.append(problem.cassini_2())
        prob_list.append(problem.gtoc_1())
        prob_list.append(problem.rosetta())
        prob_list.append(problem.messenger_full())
        prob_list.append(problem.tandem(prob_id = 6, max_tof = 10))

    algo_list = [algorithm.pso(gen = number_of_generations),
                 algorithm.de(gen = number_of_generations,xtol=1e-30, ftol=1e-30),
                 algorithm.jde(gen = number_of_generations, variant_adptv=2,xtol=1e-30, ftol=1e-30),
                 algorithm.de_1220(gen = number_of_generations, variant_adptv=2,xtol=1e-30, ftol=1e-30),
                 algorithm.sa_corana(iter = number_of_generations*number_of_individuals,Ts = 1,Tf = 0.01),
                 algorithm.ihs(iter = number_of_generations*number_of_individuals),
                 algorithm.sga(gen = number_of_generations),
                 algorithm.cmaes(gen = number_of_generations,xtol=1e-30, ftol=1e-30),
                 algorithm.bee_colony(gen = number_of_generations/2)]
                 
    print('\nTrials: ' + str(n_trials) + ' - Population size: ' + str(pop_size) + ' - Generations: ' + str(n_gen))
    for prob in prob_list:
        print('\nTesting problem: ' + prob.get_name() + ', Dimension: ' + str(prob.dimension) )
        print('With Population Size: ' +  str(pop_size) )
        for algo in algo_list:
            print(' ' + str(algo))
            best = []
            best_x = []
            for i in range(0,number_of_trials):
                isl = island(algo,prob,number_of_individuals)
                isl.evolve(1)
                isl.join()
                best.append(isl.population.champion.f)
                best_x.append(isl.population.champion.x)
            print(' Best:\t' + str(min(best)[0]))
            print(' Mean:\t' + str(mean(best)))
            print(' Std:\t' + str(std(best)))
                     
                 

def plot_best(best):
    import matplotlib
    matplotlib.use('Qt4Agg')
    
    import numpy
    import pylab
    
    #pylab.plot(t, s)
    pylab.plot(best)
    pylab.xlabel('Generation')
    pylab.ylabel('Cost')
    pylab.title('About as simple as it gets, folks')
    #pylab.yscale('log')
    pylab.grid(True)
    pylab.show()

def plot_multi(multi,names=None,trials=1):
    #import matplotlib
    #matplotlib.use('Qt4Agg')
    import numpy
    import pylab
    
    lines = [pylab.plot(line)[0] for line in multi]
    if names is None:
        pylab.legend(lines,[str(i) for i in range(len(lines))])
    else:
        pylab.legend(lines,names)
    
    pylab.xlabel('Generation')
    pylab.ylabel('Cost')
    pylab.title('Comparison of algorithms. Generation best averaged over '+str(trials)+' trials' )
    #pylab.yscale('log')
    pylab.grid(True)
    pylab.show()
    
        
def single_test(algo,toal_evals=1200, pop_size = 20):
    
    number_of_individuals = pop_size
    '''
    if(algo.get_name() in ('Particle Swarm optimization','Artificial Bee Colony optimization' )):
        number_of_generations = n_gen/2
    else:
        number_of_generations = n_gen
    '''
    
    prob = gauss_problem_stochastic(dim=3, seed=123456)
    isl = island(algo,prob,number_of_individuals)
    
    #best = []
    #best_x = []
    #for i in range(0,number_of_generations):
    while isl.problem.get_evals() < total_evals:
        isl.evolve(1)
        isl.join()
        #best.append( (isl.problem.get_evals(),isl.population.champion.f[0]) )
        #best_x.append(isl.population.champion.x)
    else:
        print isl.problem.get_evals() 
        
    #print(' Best:\t' + str(min(best)[0]))
    #print(' Mean:\t' + str(mean(best)))
    #print(' Std:\t' + str(std(best)))
    
    #print isl.population.champion.f
    #print isl.population.champion.x
    
    #print '!>',prob.human_readable_extra()
    
    #plot_best(best)

    return isl.problem.get_history()


def run(n_tri,n_gen,pop_size):
    import numpy
    
    #inital number of generations
    number_of_generations = 1
    number_of_trials = n_tri
    number_of_individuals = pop_size
    
    algo_list = [algorithm.pso(gen = number_of_generations),
                 algorithm.de(gen = number_of_generations,xtol=1e-30, ftol=1e-30),
                 algorithm.jde(gen = number_of_generations, variant_adptv=2,xtol=1e-30, ftol=1e-30),
                 algorithm.de_1220(gen = number_of_generations, variant_adptv=2,xtol=1e-30, ftol=1e-30),
                 #algorithm.sa_corana(iter = number_of_generations*number_of_individuals,Ts = 1,Tf = 0.01),
                 algorithm.ihs(iter = number_of_generations*number_of_individuals),
                 #algorithm.sga(gen = number_of_generations),
                 algorithm.cmaes(gen = number_of_generations,xtol=1e-30, ftol=1e-30),
                 algorithm.bee_colony(gen = number_of_generations)
    ]
                 
    
    algo_means_list = []
    algo_name_list = []
    print('\nTrials: ' + str(n_tri) + ' - Population size: ' + str(pop_size) + ' - Generations: ' + str(n_gen))
    for algo in algo_list:
            print(' ' + str(algo))
            algo_name_list.append(algo.get_name())
            
            trials = []
            trials_max_len = 0
            for i in range(number_of_trials):
                print ".",
                trials.append(single_test(algo,total_evals,pop_size))

            t_minlen = min([len(t) for t in trials])
            t_new = numpy.array([t[:t_minlen-1] for t in trials ])
            t_new = numpy.mean(t_new,axis=0)
            algo_means_list.append(t_new[:,1])
    
    plot_multi(algo_means_list,names=algo_name_list,trials=number_of_trials)
    
if __name__=='__main__':
    trials = 30
    total_evals = 800
    pop_size = 30
    
    run(trials,total_evals,pop_size)
