# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:31:38 2022

@author: 11481
"""

# General
import time
import copy
import warnings
import numpy as np
import pandas as pd
import random
import json

# CPLEX
import cplex
from cplex.exceptions import CplexSolverError


# A in form:(i,t,j,u):(c1,c2,d)
# others: (i,t,j,u):p

###################################
######---Small example--###########
###################################

h = NUMBER_OF_VEHICLE # Number of vehicles
q = NUMBER_OF_STAFF # Number of staff
s = 10 # Number of stations
t = 96 # Number of time interval
# dummy points
do = s
dd = s+1

# parameters
ca = 130 # amortized cost of vehicle
cy = 150 # hiring cost of staff

# ============================Test example========================================
# a = {}
# for i in range(s):
#     for j in range(s):
#         for k in range(t):
#             if i!=j:
#                 a[i,k,j,k+1] = [0.15, 0.5, 0]
#             else:
#                 a[i,k,j,k+1] = [0,0,0]
# a[(1,0,0,1)][2] = 2
# a[(2,4,1,5)][2] = 1
# a[(2,2,1,3)][2] = 1
# 
# ado = {}
# for i in range(s):
#     ado[(do,0,i,0)] = [0]
# 
# ado[(do,0,dd,0)] = [0]
# 
# add = {}
# for i in range(s):
#     add[(i,t,dd,t)] = [0]
# 
# add[(do,0,dd,0)] = [0]
# =============================================================================

a = pd.read_csv("../data/model_net_oneday_10_25.csv")
a = a.set_index(['origin','init','destination','final']).T.to_dict('list')

# Decrease number of trips
# delete_num = 0
# for k,v in a.items():
#     if v[2]!=0:
#         delete_num += v[2]
#         a[k][2] = 0
    
#     # Delete 150 trips
#     if delete_num > 150:
#         break


ado = {}
for i in range(s):
    ado[(do,0,i,0)] = [0]

ado[(do,0,dd,0)] = [0]

add = {}
for i in range(s):
    add[(i,t,dd,t)] = [0]

add[(do,0,dd,0)] = [0]

data = {"A":a, "ADO":ado, "ADD":add,  "T":t, "H":h, "Q":q, "S":s}



def getModel(data):
    '''
    Construct a CPLEX model

    Returns
    -------
    model: CPLEX model

    '''
    
    t_in = time.time()
    # Initialize the model
    model = cplex.Cplex()
    
    ##########################################
    ##### ----- OBJECTIVE FUNCTION ----- #####
    ##########################################

    # Set the objective function sense
    model.objective.set_sense(model.objective.sense.minimize)



    ##########################################
    ##### ----- DECISION VARIABLES ----- #####
    ##########################################
    
    # Batch of initial distribution variables
    objVar = []
    typeVar = []
    nameVar= []
    lb = []
    ub = []

    for i in range(data['S']):
        objVar.append(ca)
        typeVar.append(model.variables.type.integer)
        nameVar.append('h[' + str(i) + ']')
        lb.append(0)
        ub.append(data['H'])
        objVar.append(cy)
        typeVar.append(model.variables.type.integer)
        nameVar.append('q[' + str(i) + ']')
        lb.append(0)
        ub.append(data['Q'])
        
        
    model.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                        lb = [lb[i] for i in range(len(lb))],
                        ub = [ub[i] for i in range(len(ub))],
                        types = [typeVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])  
    # Batch of arc variables
    objVar = []
    typeVar = []
    nameVar= []  
    
    # Vehicle
    for v in range(data['H']):
        for node in data['A']:
            objVar.append(data['A'][node][0] - data['A'][node][1])
            typeVar.append(model.variables.type.binary)
            nameVar.append('x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
        # Vehicle dummy origin and destination
        for node in data['ADO']:
            objVar.append(0)
            typeVar.append(model.variables.type.binary)
            nameVar.append('x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')

        for node in data['ADD']:
            if node[1]!=0:
                objVar.append(0)
                typeVar.append(model.variables.type.binary)
                nameVar.append('x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
                
   
                
    # Relocator
    for q in range(data['Q']):
        for node in data['A']:
            objVar.append(data['A'][node][1])
            typeVar.append(model.variables.type.binary)
            nameVar.append('z[' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
        # Relocator dummy origin and destination
        for node in data['ADO']:
            objVar.append(0)
            typeVar.append(model.variables.type.binary)
            nameVar.append('z[' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')

        for node in data['ADD']:
            if node[1]!=0:
                objVar.append(0)
                typeVar.append(model.variables.type.binary)
                nameVar.append('z[' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')

    model.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                        types = [typeVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    print('CPLEX model: all decision variables added. N variables: %r. Time: %r'\
          %(model.variables.get_num(), round(time.time()-t_in,2)))

    # Creating a dictionary that maps variable names to indices, to speed up constraints creation
    # https://www.ibm.com/developerworks/community/forums/html/topic?id=2349f613-26b1-4c29-aa4d-b52c9505bf96
    nameToIndex = { n : j for j, n in enumerate(model.variables.get_names()) }
    
    
    
    #########################################
    ##### -------- CONSTRAINTS -------- #####
    #########################################
    

    ###################################################
    ### --- Fleet and staff constraints --- ###
    ###################################################        

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    ind = []
    co = []
    ind1 = []
    co1 = []

    for i in range(data['S']):
        ind.append(nameToIndex['h[' + str(i) + ']'])
        co.append(1.0)
        ind1.append(nameToIndex['q[' + str(i) + ']'])
        co1.append(1.0)


    indicesConstr.append(ind)
    coefsConstr.append(co)
    sensesConstr.append('L')
    rhsConstr.append(data['H'])

    indicesConstr.append(ind1)
    coefsConstr.append(co1)
    sensesConstr.append('L')
    rhsConstr.append(data['Q'])     
    
          

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
    
    ###################################################
    ### --- Vehicle flow constraints --- ###
    ###################################################        

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    for v in range(data['H']):    
        for i in range(data['S']):
            for j in range(data['T']+1):
                ind = []
                co = []
                for node in data['A']:
                    if node[0] == i and node[1] == j:
                        ind.append(nameToIndex['x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(1.0)
                    if node[2] == i and node[3] == j:
                        ind.append(nameToIndex['x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(-1.0)
                        
                for node in data['ADO']:
                    if node[2] == i and node[3] == j:
                        ind.append(nameToIndex['x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(-1.0)
                        
                for node in data['ADD']:
                    if node[0] == i and node[1] == j:
                        ind.append(nameToIndex['x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(1.0)
                        
                indicesConstr.append(ind)
                coefsConstr.append(co)
                sensesConstr.append('E')
                rhsConstr.append(0)   
                
    # Dummy origin
    for v in range(data['H']):
        ind = []
        co = []
        for node in data['ADO']:
            ind.append(nameToIndex['x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
            co.append(1.0)   
                
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(1)  
            
    # Dummy destination  
    for v in range(data['H']):
        ind = []
        co = []
        for node in data['ADD']:
            ind.append(nameToIndex['x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])   
            co.append(-1.0) 
                    
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(-1)  

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
                
    ###################################################
    ### --- Vehicle intial distribution constraints --- ###
    ###################################################  
    
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    for i in range(data['S']):
        ind = []
        co = []
        for v in range(data['H']):
            for node in data['ADO']:
                if node[2] == i:
                    ind.append(nameToIndex['x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                  + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                    co.append(1.0)
        ind.append(nameToIndex['h[' + str(i) + ']'])
        co.append(-1.0)
        
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(0)

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
    
    #######################################
    ## - User reservation constraints - ###
    #######################################
    
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    for node in data['A']:
        ind = []
        co = []
        for v in range(data['H']):
            ind.append(nameToIndex['x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                  + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
            co.append(1)
            
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('G')
        rhsConstr.append(data['A'][node][2]) #demand        

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
        
    #######################################
    ## - Staff flow constraints - ###
    #######################################
    
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    for node in data['A']:
        ind = []
        co = []
        if node[0] != node[2]:
            for q in range(data['Q']):
                ind.append(nameToIndex['z[' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                  + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                co.append(1)
            for v in range(data['H']):
                ind.append(nameToIndex['x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                  + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                co.append(-1)  
                
            indicesConstr.append(ind)
            coefsConstr.append(co)
            sensesConstr.append('G')
            rhsConstr.append(-data['A'][node][2])

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
        
    ###################################################
    ### --- Staff flow conservation --- ###
    ###################################################        

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    for q in range(data['Q']):    
        for i in range(data['S']):
            for j in range(data['T']+1):
                ind = []
                co = []
                for node in data['A']:
                    if node[0] == i and node[1] == j:
                        ind.append(nameToIndex['z[' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(1.0)
                    if node[2] == i and node[3] == j:
                        ind.append(nameToIndex['z[' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(-1.0)
                        
                for node in data['ADO']:
                    if node[2] == i and node[3] == j:
                        ind.append(nameToIndex['z[' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(-1.0)
                        
                for node in data['ADD']:
                    if node[0] == i and node[1] == j:
                        ind.append(nameToIndex['z[' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(1.0)
                        
                indicesConstr.append(ind)
                coefsConstr.append(co)
                sensesConstr.append('E')
                rhsConstr.append(0)   
                
    # Dummy origin
    for q in range(data['Q']):
        ind = []
        co = []
        for node in data['ADO']:
            ind.append(nameToIndex['z[' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
            co.append(1.0)   
                
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(1)  
            
    # Dummy destination  
    for q in range(data['Q']):
        ind = []
        co = []
        for node in data['ADD']:
            ind.append(nameToIndex['z[' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])   
            co.append(-1.0) 
                    
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(-1)  

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])

    ###################################################
    ### --- Staff intial distribution constraints --- ###
    ###################################################  
    
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    for i in range(data['S']):
        ind = []
        co = []
        for q in range(data['Q']):
            for node in data['ADO']:
                if node[2] == i:
                    ind.append(nameToIndex['z[' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                  + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                    co.append(1.0)
        ind.append(nameToIndex['q[' + str(i) + ']'])
        co.append(-1.0)
        
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(0)

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
    
    return model
    
def solveModel(data, model):
    ''' Solve the given model, return the solved model.
        Args:
            model          cplex model to solve
        Returns:
            model          cplex model solved
    '''
    try:
        #model.set_results_stream(None)
        #model.set_warning_stream(None)

        model.solve()

        ### PRINT OBJ FUNCTION
        print('Objective function value (profit): %r\n' %round(model.solution.get_objective_value(),3))
        #print('Objective function value of optimizer profit : %r\n' %round(model.solution.get_objective_value(),3), file=f)

        ### INITIALIZE DICTIONARY OF RESULTS
        results = {}
        
        results['AR'] = {str((node[0],node[1],node[2],node[3])):-demand[2] for node,demand in data['A'].items() if node[0]!=node[2]}
        results['AC'] = {str((node[0],node[1],node[2],node[3])):0 for node,demand in data['A'].items() if node[0]!=node[2]}
        results['AT'] = {str((node[0],node[1],node[2],node[3])):0 for node in data['A'] if node[0]!=node[2]}
        results['ADO'] = {}
        results['IQ'] = {}
        results['IH'] = {}

        ### SAVE RESULTS
        # Rebalancing
        for v in range(data['H']):  
            for node in data['A']:
                if node[0]!=node[2]:
                    results['AR'][str((node[0],node[1],node[2],node[3]))] += model.solution.get_values('x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
                    results['AC'][str((node[0],node[1],node[2],node[3]))] += model.solution.get_values('x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                                                  + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
        
        # Relocation
        for q in range(data['Q']):
            for node in data['A']:
                if node[0]!=node[2]:
                    results['AT'][str((node[0],node[1],node[2],node[3]))] += model.solution.get_values('z[' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
        
        for each in results['AT']:
            results['AT'][each] -= results['AR'][each]
        
        
        # Initial distribution
        for i in range(data['S']): 
            results['IH'][str(i)] = model.solution.get_values('h[' + str(i) + ']')
            results['IQ'][str(i)] = model.solution.get_values('q[' + str(i) + ']')
            
        for v in range(data['H']):
            for node in data['ADO']:
                results['ADO'][str((v,node[0],node[1],node[2],node[3]))] = model.solution.get_values('x[' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')

            
        results['profits'] = model.solution.get_objective_value()
        return results
    
    except CplexSolverError as e:
        raise Exception('Exception raised')

    return results




if __name__ == '__main__':
    
    #hour = 3
    #time_limit = 3600*hour

    t_0 = time.time()
    
    #Read instance
    #data = data_file.getData()
    t_1 = time.time()

    #SOLVE MODEL
    model = getModel(data)
    model.parameters.mip.interval.set(1)
    #model.parameters.timelimit.set(time_limit)
    results = solveModel(data, model)

    t_2 = time.time()
    
    print('\n -- TIMING -- ')
    print('Get data + Preprocess: %r sec' %(t_1 - t_0))
    print('Run the model: %r sec' %(t_2 - t_1))
    print('\n ------------ ')
    
    # Save result
    with open(f"{h}_{q}_{s}_{t}_{int(t_2 - t_1)}.json", "w") as f:
        json.dump(results,f) 

        

