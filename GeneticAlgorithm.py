#!/usr/bin/env python3
import numpy as np
import subprocess as sb
import sys,os,copy,random,datetime, math

class GeneticAl:
    '''
    Generates object with a population set and all data.
    Input: <input data file name>, [<Size of population>], [<Number of generations>]
    Attributes:
    init
    '''

    def __init__(self,data_file,N_pop=500,N_gen=5000):
        
        self.filename=data_file
        with open(self.filename) as f:
            data_=f.readlines()
        self.N_elite=0
        ind_elite=len(data_) #index where elite starts, put to end of file if oonly Input
        for i in range(len(data_)):
            if data_[i].lower().startswith('elite'):
                self.N_elite=int(data_[i].split()[1])
                ind_elite=i
                break
        self.N_pop=N_pop
        self.N_gen=N_gen
        self.input_=data_[:ind_elite]
        for i in range(len(self.input_)):
            if self.input_[i].lower().startswith('d'):
                ind_dim=i
            if self.input_[i].lower().startswith('o'):
                ind_obj=i
        self.N_dim=int(self.input_[ind_dim].split()[1])
        self.N_obj=int(self.input_[ind_obj].split()[1])
        
        #Reading input
        print (" Reading input")
        try:
            # parameters for Dimension: <Min> <Max> <Increment>
            self.P_dim=np.zeros((self.N_dim,3))
            self.P_dim[:]=np.array([list(map(float,x.split()[:3])) for x in self.input_[ind_dim+1:ind_dim+self.N_dim+1]])
            
            # parameters for Objective: <Type> <Target> <Margin> <Tier>
            P_obj=np.zeros((self.N_obj,4))
            for i in range(self.N_obj):
                line=self.input_[ind_obj+1+i].split()
                P_obj[i][0]=(-1 if line[0].lower() == "max" else (1 if line[0].lower() == "min" else 0))
                P_obj[i][1]=(float(line[0]) if P_obj[i][0] == 0  else 0)
                P_obj[i][2:4]=np.array(list(map(float,line[1:3])))
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
                if self.exist(self.pop[indx],range(indx)):
                    self.pop[indx][:]=self.pop[self.N_elite-1]
                    self.N_elite-=1
                else:
                    indx+=1
            print (" {0} elites found".format(self.N_elite))
                
        print (" Completing generation of random initial population")
        self.init_pop(range(self.N_elite,self.N_pop))
        
        print (" Calculating objectives")
        self.calc_obj(indx=range(self.N_elite,self.N_pop))
        
        print (" Ranking population")
        #self.rank_pop()
        
        print (" Saving Elite (First Pareto set)")
        #self.save_things()
        
    def init_pop(self,indx=None):
        if indx is None: indx=range(self.N_pop)
        for i in indx:
            max_loop=200
            for loop in range(max_loop):
                hold=np.array([ (random.uniform(self.P_dim[j][0],self.P_dim[j][1]) if self.P_dim[j][2]==0 else self.P_dim[j][0]+self.P_dim[j][2]*random.randint(0,int((self.P_dim[j][1]-self.P_dim[j][0])/float(self.P_dim[j][2])))) for j in range(self.N_dim)])
                # Check if hold exists in self.pop, excluding, itself and all indx not initialised yet
                if not self.exist(hold,list(set(range(self.N_pop))-set(indx[indx.index(i):]))):
                    self.pop[i][:self.N_dim]=hold
                    break
            if loop == max_loop-1:
                raise RuntimeError(" Houston we got a problem: It seems the Dimension requirements don't allow random {0} population easily".format(self.N_pop))
          
    def calc_obj(self,param=None,indx=None):
        '''
        Calculates objective for passed parameter or self.pop for index provided
        param
            numpy array of size of dimension (or population set starting with dimensions)
        indx, default= range(<Total population count>)
            list of indixes of population to calculate objective for
        Note: Always sort the population on rank after implementation; self.rank_pop()
        '''
        try:
            if param is None:
                if indx is None: indx=range(self.N_pop)
                for i in indx:
                        out=sb.run(['./run.sh']+list(map(str,self.pop[i][:self.N_dim])),stdout=sb.PIPE).stdout.decode('utf-8')
                        self.pop[i][-1*self.N_obj:]=np.array(list(map(float,out.split())))
                        #if not ranked, N_elite is trivial
                        if not self.ranked:
                            self.N_elite=i
                        self.save_things()
                            
            else:
                if param.size < self.N_dim:
                    print ("Input numpy array size < N_dim(={0})".format(self.N_dim))
                    return None
                out=sb.run(['./run.sh']+(list(map(str,param[:self.N_dim]))),stdout=sb.PIPE).stdout.decode('utf-8')
                param_out=np.zeros(self.N_dim+self.N_obj)
                param_out[:self.N_dim]=param[:self.N_dim]
                param_out[-1*self.N_obj:]=np.array(list(map(float,out.split())))
                return param_out
        except ValueError as error:
            with open("Calc_error.log",'a') as f:
                f.write('## '+str(datetime.datetime.now())+'\n')
                f.write("Input:"+"\n"+" ".join(list(map(str,param.tolist())))+"\n")
                f.write("Output:"+"\n"+out+"\n")
            raise ValueError("::Error in calc_obj; check Calc_error.log::"+error.args[0])
    '''

    def calc_obj(self,param):
        try:
            if param.size < self.N_dim:
                print ("Input numpy array size < N_dim(={0})".format(self.N_dim))
                return None
            out=sb.run(['./run.sh']+(list(map(str,param[:self.N_dim]))),stdout=sb.PIPE).stdout.decode('utf-8')
            param_out=np.zeros(self.N_dim+self.N_obj)
            param_out[:self.N_dim]=param[:self.N_dim]
            param_out[-1*self.N_obj:]=np.array(list(map(float,out.split())))
            return param_out
        except ValueError as error:
            with open("Calc_error.log",'a') as f:
                f.write('## '+str(datetime.datetime.now())+'\n')
                f.write("Input:"+"\n"+" ".join(list(map(str,param.tolist())))+"\n")
                f.write("Output:"+"\n"+out+"\n")
            raise ValueError("::Error in calc_obj; check Calc_error.log::"+error.args[0])
    '''

    def rank_pop(self,param=None):
        tier={}
        if param is None:
            for i in range(self.N_pop):
                pass
        if not self.ranked: self.ranked=True

    def mutate():
        pass

    def exist(self,param,indx=None):
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
            
    def save_things(self):
        '''
        Saves the input parameters and Elite people
        Also during intialising saves un-ranked population
        as an backup if something fails
        Also keeps un-elites (and un-initialised, when data not ranked) after commented line
        (important when calc_obj raises error while in use after initialisation)
        '''
        f=open(self.filename,'w+')
        f.write(''.join(self.input_))
        f.write('\nELITE {0} (Data {1}ranked)\n'.format(self.N_elite,('' if self.ranked else 'not ')))
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
    
    # parameters for Objective: <Type> <Target> <Margin> <Tier>
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
    N_gen=5000
    
    #Supplantation: Supplanting lower ranked population with random population to increase
    #possibility to reach global minima. This is done when:
    #I: First Pareto set grows large, i.e. a huge higher rank falls in a minima
    #II: After certain generations
    Pareto_max_ratio=0.2 #the max size ratio for first pareto to initiate supplantation
    Pop_purge_ratio=0.2 #the size ratio of bottom ranked population to be supplanted
    Purge_check_raio=0.1 #number ratio of N_gen to jump while checking supplantation condition   
    
    #f=open('GA.log','a')
    #f.write('## Attempt :: '+str(datetime.datetime.now())+'\n')
    
    arg=sys.argv[1:]
    
    data_file=arg[0]
    
    if not os.path.isfile(data_file):
        print(" //ERROR: "+data_file+" file not found")
        #f.write(" //ERROR: "+data_file+" file not found"+"\n")
        sys.exit()
    
    pop=GeneticAl(data_file,N_pop,N_gen)
        
        

        
        
        
        
        
            
