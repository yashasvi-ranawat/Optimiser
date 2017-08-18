#!/usr/bin/env python3
import numpy as np
import subprocess as sb
import sys,os,copy,random,datetime, math


def CalcObjDefault(param):
    '''
    Calculates objective for passed parameter using run.sh
    arguments:
        numpy array of size of dimension
    returns numpy array of size of objective
    Note: To make your own custom function:
    1. define the function which takes in a numpy array of size (<>.N_dim,)
    Uses the <>.N_dim indices to give <>.N_obj 
    and passes numpy array of size (<>.N_obj,)
    This way, all other functions can access it.
    2. <>.func=<Custom Function>
    '''
    try:
        out=sb.run(['./run.sh']+(list(map(str,param))),stdout=sb.PIPE).stdout.decode('utf-8')
        param_out=np.array(list(map(float,out.split())))
    except ValueError as error:
        with open("Calc_error.log",'a') as f:
            f.write('## '+str(datetime.datetime.now())+'\n')
            f.write("Input:"+"\n"+" ".join(list(map(str,param[:self.N_dim].tolist())))+"\n")
            f.write("Output:"+"\n"+out+"\n")
        raise ValueError("Error in CalcObjDefault; check Calc_error.log:"+error.args[0])
    return param_out

class Optim:
    '''
    Generates object with a population set and all data.
    Input: <input data file name>,
            [algo=<Type of algorithm, default="GeneticAl">], 
            [N_pop=<Size of population, default=500 for GeneticAl and No. of dim +1 for Simplex>]
            [func=<Custom function>]
    Custom function, default=CalcObjDefault (The one that uses run.sh)
    To make your own custom function:
    1. define the function which takes in an object and a numpy array of size (<>.N_dim,)
    Use the <>.N_dim indices to give <>.N_obj
    and pass numpy array of size (<>.N_obj,)
    This way, all other functions can access it.
    2. <>.func=<Custom Function> (__init__ does it automatically if function passed)
    Attributes:
    InitPop
    '''
    
    algo_list=["GeneticAl","Simplex"]
    
    def __init__(self,data_file,algo="GeneticAl",N_pop=None,func=CalcObjDefault):
        
        self.algo=algo
        self.func=func
        
        if not self.algo in self.algo_list:
            print(" {} not supported \n Supported Algorithms: {}".format(algo,algo_set))
            return None
        
        self.filename=data_file
        with open(self.filename) as f:
            data_=f.readlines()
        self.N_elite=0
        ind_elite=len(data_) #index where elite starts, put to end of file if only Input
        for i in range(len(data_)):
            if data_[i].strip().lower().startswith('elite'):
                self.N_elite=int(data_[i].split()[1])
                ind_elite=i
                break
        self.input_=data_[:ind_elite]
        for i in range(len(self.input_)):
            if self.input_[i].strip().lower().startswith('d'):
                ind_dim=i
            if self.input_[i].strip().lower().startswith('o'):
                ind_obj=i
        self.N_dim=int(self.input_[ind_dim].split()[1])
        self.N_obj=int(self.input_[ind_obj].split()[1])
        
        if self.algo == self.algo_list[1]: #for Simplex
            self.N_pop=self.N_dim +1
        if self.algo == self.algo_list[0]: #for GeneticAl
            if N_pop is None: N_pop=500 #Default
            self.N_pop=N_pop
        
        #Reading input
        print (" Reading input")
        try:
            # parameters for Dimension: <Min> <Max> <Increment>
            self.P_dim=np.zeros((self.N_dim,3))
            self.P_dim[:]=np.array([list(map(float,x.split()[:3])) for x in self.input_[ind_dim+1:ind_dim+self.N_dim+1]])
            
            # parameters for Objective: <Type> <Target> <Tolerance> <Tier>
            #P_obj:
            #[0] :: Type = 1,-1,0 for min, max, target respectively
            #[1] :: Target= target value, else 0 for max and min type
            #[2] :: Tolerance
            P_obj=np.zeros((self.N_obj,3))
            #Tier, dictionary of tiers as keys and list of index for corresponding tier as their values
            self.tier={}
            for i in range(self.N_obj):
                line=self.input_[ind_obj+1+i].split()
                P_obj[i][0]=(-1 if line[0].lower() == "max" else (1 if line[0].lower() == "min" else 0))
                P_obj[i][1]=(float(line[0]) if P_obj[i][0] == 0  else 0)
                P_obj[i][2]=float(line[1])
                tier_i=float(line[2])
                self.tier[tier_i]=self.tier.get(tier_i,[])+[i]
            self.P_obj=P_obj
        except ValueError as error:
            raise ValueError(":Error while reading input data:"+error.args[0])
        
        # Initialise population
        self.pop=np.zeros((self.N_pop,self.N_dim+self.N_obj))
        self.ranked=False #Data not ranked
        
        # Reading elite pop

        if not self.N_elite==0:
            print (" Reading Elite data")
            # if more elite data then choose first N_pop
            if self.N_elite > self.N_pop:
                self.N_elite=self.N_pop
            try:
                self.pop[:self.N_elite]=np.array([list(map(float,x.split()[:self.N_dim+self.N_obj])) for x in data_[ind_elite+1:ind_elite+self.N_elite+1]])
            except ValueError as error:
                raise ValueError(":Error while reading elite data:"+error.args[0])
            #Removing duplicates
            indx=1
            while indx < self.N_elite:
                if self.Exist(self.pop[indx],range(indx)):
                    self.pop[indx][:]=self.pop[self.N_elite-1]
                    self.N_elite-=1
                else:
                    indx+=1
            print (" {} elites found".format(self.N_elite))
        
        #Adjusting N_elite to contain entire population for Simplx
        #Facilitates easy saving with same functions for genetic algo
        #if self.algo == self.algo_list[1]:
        #    self.N_elite=self.N_pop
        
        print (" Completing generation of random initial population")
        self.InitPop(range(self.N_elite,self.N_pop))
        
        print (" Calculating objectives")
        self.CalcObjIndx(range(self.N_elite,self.N_pop))
        
        print (" Ranking population")
        self.RankPop()
        
        #Update N_elite, if GeneticAl
        if self.algo == self.algo_list[0]:
            self.UpdateElite()
        
        print (" Saving Elite")
        self.SaveThings()
        
    def InitPop(self,indx=None):
        '''
        Initialises population for index (list) provided
        If nothing is passed, entire population is initialised
        '''
        if indx is None: indx=range(self.N_pop)
        for i in indx:
            max_loop=200
            for loop in range(max_loop):
                hold=np.array([ (random.uniform(self.P_dim[j][0],self.P_dim[j][1]) if self.P_dim[j][2]==0 else self.P_dim[j][0]+self.P_dim[j][2]*random.randint(0,divmod(self.P_dim[j][1]-self.P_dim[j][0],self.P_dim[j][2])[0])) for j in range(self.N_dim)])
                # Check if hold exists in self.pop, excluding, itself and all indx not initialised yet
                if not self.Exist(hold,list(set(range(self.N_pop))-set(indx[indx.index(i):]))):
                    self.pop[i][:self.N_dim]=hold
                    break
            if loop == max_loop-1:
                raise RuntimeError(" Houston we got a problem: It seems the Dimension requirements don't allow random {} population easily".format(self.N_pop))
          
    def CalcObjIndx(self,indx=None):
        '''
        Calculates objective for self.pop for index provided
        argument:
        indx, default= range(<Total population count>)
            list of indixes of population to calculate objective for
        Note: rank the population after implementation; <>.RankPop()
        '''
        if indx is None: indx=range(self.N_pop)
        for i in indx:
            if os.path.isfile('STOP'): sys.exit() #Failsafe
            self.pop[i]=self.CalcObj(self.pop[i])
            #if not ranked, N_elite is trivial, updated to save all CalcObj calculated population
            if not self.ranked:
                self.N_elite=i+1
            #Population not ranked since it is used for initialisation; population with objective missing exist
            #self.RankPop()
            self.SaveThings()
            
    def CalcObj(self,param):
        '''
        Calculates objective for passed parameter
        arguments:
            numpy array of size of dimension (or population set starting with dimensions)
        Note: To make your own custom function:
        1. define the function which takes in a numpy array of size (<>.N_dim,)
        Uses the <>.N_dim indices to give <>.N_obj
        and passes numpy array of size (<>.N_obj,)
        This way, all other functions can access it.
        2. <>.func=<Custom Function>
        '''
        if param.size < self.N_dim:
            raise ValueError("Input numpy array size < N_dim(={})".format(self.N_dim))
        param_out=np.zeros(self.N_dim+self.N_obj)
        param_out[:self.N_dim]=param[:self.N_dim]
        try:
            hold=self.func(param[:self.N_dim])
            #Changing target based objective to abs(error)
            for j in range(self.N_obj):
                #if <Type> ==0, replace obj with abs(obj-<Target>)
                if self.P_obj[j][0]==0:
                    hold[j]=abs(hold[j]-self.P_obj[j][1])
            param_out[-1*self.N_obj:]=hold
            return param_out
        except ValueError as error:
            with open("Calc_error.log",'a') as f:
                f.write('## '+str(datetime.datetime.now())+'\n')
                f.write("Input:"+"\n"+" ".join(list(map(str,param[:self.N_dim].tolist())))+"\n")
                f.write("Output:"+"\n"+out+"\n")
            raise ValueError("Error in <>.CalcObj; check Calc_error.log:"+error.args[0])
    
    def SuperiourityOrderIs(self,param1=None,param2=None):
        '''
        Based on objective requirements
        Returns True if param1 is superiour than param2
                False if param1 is not superiour than param2 (i.e. param2 is superiour or equal to param1)
        Uses hard ranking (Pareto method), i.e. ALL (not maximum) objectives for same tier should be superiour
        '''
        if param1 is None and param2 is None:
            raise TypeError("SuperiourityOrderIs() takes 2 arguments")
        elif param1.size != self.N_dim+self.N_obj and param2.size != self.N_dim+self.N_obj:
            raise ValueError("param1 and param2 should be numpy array of size {}, but {} and {} were given,respectively".format(self.N_dim+self.N_obj,param1.size,param2.size))
        
        obj1=param1[-1*self.N_obj:]
        obj2=param2[-1*self.N_obj:]
        for i in sorted(list(self.tier.keys())):
            #for each tier list, we use hard ranking (Pareto method), i.e. ALL (not maximum) objectives in same tier should be superiour 
            x=0
            for j in self.tier[i]:
                #if obj1's j th objective is not superiour to obj2's, then break
                if not (obj2[j]-obj1[j])*(-1 if self.P_obj[j][0]==-1 else 1) > self.P_obj[j][2]: 
                    break
                x+=1
            if x==len(self.tier[i]): return True #if the loop above didn't break prematurely, then obj1 is superiour
            #Here, obj2 is either equal or superiour to obj1
            #Checking if obj1 is equal to obj2, only then heading to tier below
            x=0
            for j in self.tier[i]:
                #if obj2's j th objective is not superiour to obj1's, then break
                if not (obj1[j]-obj2[j])*(-1 if self.P_obj[j][0]==-1 else 1) > self.P_obj[j][2]: 
                    break
                x+=1
            if x==len(self.tier[i]): return False #if the loop above didn't break prematurely, then obj2 is superiour
            #Here, obj1 is equal to obj2
            #Heading to tier below
        #Here, all tiers are checked and obj1 is still equal to obj2
        return False

    def RankPop(self,param=None):
        '''
        Ranks parameters provided, if none, then ranks entire population
        '''
        #if nothing is passed to function, then rank whole population and return nothing
        if param is None: 
            param=np.zeros((self.N_pop,self.N_dim+self.N_obj))
            param[:]=self.pop
            return_=False
        else: #else return param
            return_=True
        
        #Making a list of indices for easy ranking
        #The indices will be ranked, rather than the whole numpy array
        #Then the new numpy array will be shuffled on this
        rank_arg=list(range(param.shape[0]))
        
        for i in range(1,param.shape[0]):
            for j in range(i):
                #if i'th pop is superior than j'th pop, then insert i'th index at j position 
                if self.SuperiourityOrderIs(self.pop[i],self.pop[rank_arg[j]]):
                    rank_arg.insert(j,rank_arg.pop(i))
                    break
        
        if return_:
            #Making ranked param
            param_ranked=np.zeros(param.shape)
            for i in range(param.shape[0]):
                param_ranked[i]=param[rank_arg[i]]
            return param_ranked
        else:
            if not self.ranked: self.ranked=True
            for i in range(param.shape[0]):
                self.pop[i]=param[rank_arg[i]]
        
    def UpdateElite(self):
        '''
        Updates N_elite to count of first pareto
        '''
        param=np.zeros((2,self.N_dim+self.N_obj))
        for i in range(1,self.N_pop):
            #if [i-1] is clear superiour than [i], then index of last elite is i-1; hence No. of elites=i
            if self.SuperiourityOrderIs(self.pop[i-1],self.pop[i]):
                self.N_elite=i
                return None
    
    def SimplexCrawl(self,iter_=1,alpha=1,gamma=2,rho=0.5,sigma=0.5):
        '''
        Makes the simplex crawl using Nelder-Mead method
        Parameters passed: 
        iter_, default=1, number of crawls
        alpha, default=1, reflection coefficients
        gamma, default=2, expansion coefficients
        rho, default=0.5, contraction coefficients
        sigma, default=0.5, shrink coefficients
        
        Refrences:
        https://en.wikipedia.org/wiki/Nelder-Mead_method
        McKinnon, K.I.M. "Convergence of the Nelder-Mead simplex method to a non-stationary point". SIAM J Optimization
        '''
        if self.algo !=self.algo_list[1]:
            raise RuntimeError("{} algorithm does not work with method: SimplexCrawl".format(self.algo))
        #Asserting proper parameters
        assert alpha>0,"alpha > 0, given alpha={}".format(alpha)
        assert gamma>1,"gamma > 1, given gamma={}".format(gamma)
        assert 0<rho<=0.5,"0<rho<=0.5, given rho={}".format(rho)
        assert sigma>0,"sigma > 0, given sigma={}".format(sigma)
        
        centroid=np.zeros(self.N_dim+self.N_obj)
        reflection=np.zeros(self.N_dim+self.N_obj)
        expansion=np.zeros(self.N_dim+self.N_obj)
        contraction=np.zeros(self.N_dim+self.N_obj)
        param=np.zeros((2,self.N_dim+self.N_obj)) #for comparing
        
        for ii in range(iter_):
            if os.path.isfile('STOP'): sys.exit() #Failsafe
            centroid[:]=np.sum(self.pop[:-1],axis=0)/(self.N_pop-1)
            #Initialising required points
            #reflection point, modelled to Dimension requirements
            reflection=self.RoundToDim(centroid+alpha*(centroid-self.pop[-1]))
            
            #Calculating objective for corresponding point
            reflection[:]=self.CalcObj(reflection)
            #print ("reflection:{}".format(reflection))
            
            if not self.SuperiourityOrderIs(self.pop[0],reflection): #reflection is better or equal to rank 0, testing expansion
                #expansion point, modelled to Dimension requirements
                expansion=self.RoundToDim(centroid+gamma*(reflection-centroid))
                #Calculating objective for corresponding point
                expansion[:]=self.CalcObj(expansion)
                #print ("expansion:{}".format(expansion))
                
                if not self.SuperiourityOrderIs(self.pop[0],expansion): #Expansion usefull
                    #Take expansion to best rank 0, so the simplex can crawl on plateaus
                    self.pop[-1]=self.pop[0] #Ranking at end will take care
                    self.pop[0]=expansion
                
                else: #reflection usefull
                    #Take reflection to best rank 0, so the simplex can crawl on plateaus
                    self.pop[-1]=self.pop[0] #Ranking at end will take care
                    self.pop[0]=reflection
                
            else: #reflection is not best, checking if reflection is useful
            
                if self.SuperiourityOrderIs(reflection,self.pop[-2]): #reflection usefull
                    self.pop[-1]=reflection
                    
                else: #reflection not usefull, testing contraction
                
                    if self.SuperiourityOrderIs(reflection,self.pop[-1]): #Outside Contraction case
                        
                        #contraction point, modelled to Dimension requirements
                        contraction=self.RoundToDim(centroid+rho*(centroid-self.pop[-1]))
                        #Calculating objective for corresponding point
                        contraction[:]=self.CalcObj(contraction)
                        #print ("out contraction:{}".format(contraction))
                        
                        if not self.SuperiourityOrderIs(reflection,contraction): #Contraction usefull
                            self.pop[-1]=contraction
                            
                        else: #contraction not useful, shrinking
                            for i in range(1,self.N_pop):
                                self.pop[i]=self.RoundToDim(self.pop[i]+sigma*(self.pop[0]-self.pop[i]))
                            self.CalcObjIndx(range(1,self.N_pop))
                                
                    else: #Inside contraction case
                    
                        #contraction point, modelled to Dimension requirements
                        contraction=self.RoundToDim(centroid+rho*(self.pop[-1]-centroid))
                        #Calculating objective for corresponding point
                        contraction[:]=self.CalcObj(contraction)
                        #print ("in contraction:{}".format(contraction))
                        
                        if self.SuperiourityOrderIs(contraction,self.pop[-1]): #Contraction usefull
                            self.pop[-1]=contraction
                            
                        else: #contraction not useful, shrinking
                            for i in range(1,self.N_pop):
                                self.pop[i]=self.RoundToDim(self.pop[i]+sigma*(self.pop[0]-self.pop[i]))
                            self.CalcObjIndx(range(1,self.N_pop))
                                
            #Ranking and saving
            self.RankPop()
            self.SaveThings()
    
    def RoundToDim(self,param=None):
        '''
        Rounds the dimenions of given param to dimension requirements, while also limiting to required space
        Also, the returned parameter has all objectives reinitialised to zero
        '''
        if param is None:
            raise TypeError("Takes one parameter")
        elif param.size < self.N_dim:
            raise ValueError("size of parameter less than N_dim (= {})".format(self.N_dim))
        for i in range(self.N_dim):
            #Checking if not lower than min
            param[i]=max(self.P_dim[i][0],param[i])
            #Checking if not higher than max
            param[i]=min(self.P_dim[i][1],param[i])
            #modelling to Dimension requirements
            if self.P_dim[i][2] != 0:
                param[i]=round((param[i]-self.P_dim[i][0])/self.P_dim[i][2])*self.P_dim[i][2]+self.P_dim[i][0]
        param_out=np.zeros(self.N_dim+self.N_obj)
        param_out[:self.N_dim]=param[:self.N_dim]
        return param_out
            
        
    def Mutate(self,iter_=1):
        if self.algo !=self.algo_list[0]:
            raise RuntimeError("{} algorithm does not work with method: Mutate".format(self.algo))
        if self.N_pop<2:
            raise ValueError("Number of population (={}) not enough to choose two parents".format(self.N_pop))
        parent=[0,0]
        for ii in range(iter_):
            if os.path.isfile('STOP'): sys.exit() #Failsafe
            #Searching for two random unidentical parents
            while True:
                parent[0]=int((1-np.power(np.random.random(),1/2.5))*(self.N_pop-1))
                parent[1]=int((1-np.power(np.random.random(),1/2.5))*(self.N_pop-1))
                if parent[0] != parent[1]:
                    parent.sort
                    break
            #making a numpy array to hold random offspring between two parents
            param=self.pop[parent[0]]+np.random.random(self.N_dim+self.N_obj)*(self.pop[parent[1]]-self.pop[parent[0]])
            #Rounding  to Dimensional requirements
            param[:]=self.RoundToDim(param)
            #Calculating objective
            param[:]=self.CalcObj(param)
            
            #if the offspring is superiour than inferiour of the parents
            if self.SuperiourityOrderIs(param,self.pop[parent[1]]):
                #Change inferiour of the parents with child and rank whole thing
                self.pop[parent[1]]=param
                self.RankPop()
                #finding first pareto
                self.UpdateElite()
                self.SaveThings()
        

    def Exist(self,param,indx=None):
        '''
        Checks if given parameter exists in population
        param
            numpy array of parameters to be checked
        indx, default = range(<Total population count>) 
            list of index of those cases in total to be checked
            important when initialising and most population is zeros
        '''
        if indx is None: indx=range(self.N_pop)
        if param.size < self.N_dim:
            print ("Input numpy array size < N_dim(={0})".format(self.N_dim))
            return None
        for i in indx:
            if not np.linalg.norm(param[:self.N_dim]-self.pop[i][:self.N_dim]):
                return True
        return False
            
    def SaveThings(self):
        '''
        Saves the input parameters and Elite people
        Also during intialising saves un-ranked population
        as an backup if something fails
        Also keeps un-elites (and un-initialised, when data not ranked) after commented line
        (important when CalcObj raises error while in use after initialisation)
        '''
        f=open(self.filename,'w+')
        f.write(''.join(self.input_))
        f.write('ELITE {} (Data {}ranked)\n'.format(self.N_elite,('' if self.ranked else 'not ')))
        #Saving elite first
        for i in range(self.N_elite):
            f.write(' '.join(map(str,self.pop[i].tolist()))+'\n')
        #Saving rest
        if self.N_elite<self.N_pop:
            f.write('#Rest population (Could be un-itialised if data not ranked)\n')
            f.write('#if restart with following, remove commented lines and bump Elite count appropriately \n')
            for i in range(self.N_elite,self.N_pop):
                f.write(' '.join(map(str,self.pop[i].tolist()))+'\n')
        f.close()
            

'''        
def parameter_in(input_):
    for i in range(len(input_)):
        if input_[i].lower().startswith('d'):
            ind_dim=i
        if input_[i].lower().startswith('o'):
            ind_obj=i
    N_dim=int(input_[ind_dim].split()[1])
    N_obj=int(input_[ind_obj].split()[1])
    
    # parameters for Dimension: <Min> <Max> <Increment>
    P_dim=np.array([list(map(float,x.split()[:3])) for x in input_[ind_dim+1:ind_dim+N_dim+1]])
    
    # parameters for Objective: <Type> <Target> <Tolerance> <Tier>
    P_obj=np.zeros((N_obj,4))
    for i in range(N_obj):
        line=input_[ind_obj+1+i].split()
        P_obj[i][0]=(-1 if line[0].lower() == "max" else (1 if line[0].lower() == "min" else 0))
        P_obj[i][1]=(float(line[0]) if P_obj[i][0] == 0  else 0)
        P_obj[i][2:4]=np.array(list(map(float,line[1:3])))
    
    return [N_dim,P_dim,N_obj,P_obj]
'''
        
if __name__ == "__main__":
    
    #Parameters
    N_pop=500
    N_iter=5000
    
    #Supplantation: Supplanting lower ranked population with random population to increase
    #possibility to reach global minima. This is done when:
    #I: First Pareto set grows large, i.e. a huge higher rank falls in a minima
    #II: After certain generations
    Pareto_max_ratio=0.2 #the max size ratio for first pareto to initiate supplantation
    Pop_purge_ratio=0.2 #the size ratio of bottom ranked population to be supplanted
    Purge_check_raio=0.1 #number ratio of N_iter to jump while checking supplantation condition   
    
    #f=open('GA.log','a')
    #f.write('## Attempt :: '+str(datetime.datetime.now())+'\n')
    
    arg=sys.argv[1:]
    
    data_file=arg[0]
    
    if not os.path.isfile(data_file):
        print(" //ERROR: "+data_file+" file not found")
        #f.write(" //ERROR: "+data_file+" file not found"+"\n")
        sys.exit()
    
    pop=GeneticAl(data_file,N_pop)
        
        

        
        
        
        
        
            
