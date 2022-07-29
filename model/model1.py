# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 10:40:59 2022

@author: 11481
"""

# General
import time
import copy
import warnings
import numpy as np
import pandas as pd
import json
import random

# CPLEX
import cplex
from cplex.exceptions import CplexSolverError


h = NUMBER_OF_VEHICLE # Number of vehicles
q = NUMBER_OF_STAFF # Number of relocators
s = 10 # Number of stations
t = 96 # Number of time interval
b = 1 # Number of maximum relocators in one vehicle
c = [10]*s # Station capacity

# AC in form:(i,t,j,u):(d,p)
# others: (i,t,j,u):p

# =========================Test example========================================
# h = 2 # Number of vehicles
# q = 1 # Number of relocators
# s = 3 # Number of stations
# t = 5 # Number of time interval
# b = 1 # Number of maximum relocators in one vehicle
# c = [5]*s # Station capacity
# ac = {(1,0,0,1):(2,1),(2,4,1,5):(1,1),(2,2,1,3):(1,1)}
# aw = {}
# for i in range(s):
#     for j in range(t):
#         aw[(i,j,i,j+1)] = 0
# ar = {}
# for i in range(s):
#     for j in range(s):
#         for k in range(t):
#             if i!=j and k+1<=t:
#                 ar[i,k,j,k+1] = [0.5]
# at = {}
# for i in range(s):
#     for j in range(s):
#         for k in range(t):
#             if i!=j and k+1<=t:
#                 at[i,k,j,k+1] = [0.2]
# =============================================================================


###################################
###################################

ac = pd.read_csv("../data/model1_3/customer_oneday_10_25.csv")
ac = ac.set_index(['origin','init','destination','final']).T.to_dict('list')

aw = pd.read_csv("../data/model1_3/wait_oneday_10.csv")
aw = aw.set_index(['origin','init','destination','final']).T.to_dict('list')

ar = pd.read_csv("../data/model1_3/rebalance_oneday_10.csv")
ar = ar.set_index(['origin','init','destination','final']).T.to_dict('list')

at = pd.read_csv("../data/model1_3/transfer_oneday_10.csv")
at = at.set_index(['origin','init','destination','final']).T.to_dict('list')


data = {"AC":ac, "AW":aw, "AR":ar, "AT":at, "C":c, \
        "T":t, "H":h, "Q":q, "S":s}
    


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
    model.objective.set_sense(model.objective.sense.maximize)



    ##########################################
    ##### ----- DECISION VARIABLES ----- #####
    ##########################################
    
    # Batch of vehicle arc variables
    typeVar = []
    nameVar= []

    for v in range(data['H']):
        for node in data['AW']:
            typeVar.append(model.variables.type.binary)
            nameVar.append('x[w][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
        
    model.variables.add(types = [typeVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])
    
    
    # Batch of relocator arc variables
    typeVar = []
    nameVar= []
                
    for q in range(data['Q']):
        for node in data['AW']:
            typeVar.append(model.variables.type.binary)
            nameVar.append('y[w][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
                
    for q in range(data['Q']):
        for node in data['AR']:
            typeVar.append(model.variables.type.binary)
            nameVar.append('y[r][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
                
    model.variables.add(types = [typeVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    
    # Customer request and relocation
    objVar = []
    typeVar = []
    nameVar = []
    
    for v in range(data['H']):
        for node in data['AC']:
            objVar.append(data['AC'][node][1])
            typeVar.append(model.variables.type.binary)
            nameVar.append('x[c][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')    
        for node in data['AR']:
            objVar.append(-data['AR'][node][0])
            typeVar.append(model.variables.type.binary)
            nameVar.append('x[r][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
    for q in range(data['Q']):
        for node in data['AT']:
            objVar.append(-data['AT'][node][0])
            typeVar.append(model.variables.type.binary)
            nameVar.append('y[t][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
                
    # Objective function
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
    ### --- Demand constraints --- ###
    ###################################################    
    
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    # Supply does not exceed demand
    for node in data['AC']:
        ind = []
        co = []
        for v in range(data['H']):
            ind.append(nameToIndex['x[c][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
            co.append(1.0)
            
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('L')
        rhsConstr.append(data['AC'][node][0]) #demand

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
    
    
    #######################################
    ## - Station Capacity constraints - ###
    #######################################
    
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    # From time interval 1 to T-1
    for i in range(data['S']):
        for j in range(1, data['T']):
            ind = []
            co = []
            for v in range(data['H']):
                for node in data['AW']:
                    if node[0] == i and node[1] == j:
                        ind.append(nameToIndex['x[w][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(1.0)
                    
            indicesConstr.append(ind)
            coefsConstr.append(co)
            sensesConstr.append('L')
            rhsConstr.append(data['C'][i]) # capacity
    
    # At time 0
    for i in range(data['S']):
        ind = []
        co = []
        for v in range(data['H']):
            # Waiting arc
            for node in data['AW']:
                if node[0] == i and node[1] == 0:
                    ind.append(nameToIndex['x[w][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                    co.append(1.0)
                    
            # Customer arc
            for node in data['AC']:
                if node[0] == i and node[1] == 0:
                    ind.append(nameToIndex['x[c][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                    co.append(1.0)
                    
            # Relocation arc
            for node in data['AR']:
                if node[0] == i and node[1] == 0:
                    ind.append(nameToIndex['x[r][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                    co.append(1.0)
                
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('L')
        rhsConstr.append(data['C'][i]) # capacity

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])



    #######################################
    #### - Vehicle flow conservation - ####
    #######################################
    
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    
    # Each vehicle departs only from one station at time 0
    for v in range(data['H']):
        ind = []
        co = []
        for i in range(data['S']):
            # Waiting arc
            for node in data['AW']:
                if node[0] == i and node[1] == 0:
                    ind.append(nameToIndex['x[w][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                    co.append(1.0)
                    
            # Customer arc
            for node in data['AC']:
                if node[0] == i and node[1] == 0:
                    ind.append(nameToIndex['x[c][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                    co.append(1.0)
                    
            # Relocation arc
            for node in data['AR']:
                if node[0] == i and node[1] == 0:
                    ind.append(nameToIndex['x[r][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                    co.append(1.0)
                
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(1.0)
    
    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
    
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    # Vehicle conservation for all nodes
    for i in range(data['S']):
        for v in range(data['H']):
            for j in range(1, data['T']):
                ind = []
                co = []
                # Waiting arc
                for node in data['AW']:
                    if node[0] == i and node[1] == j:
                        ind.append(nameToIndex['x[w][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(1.0)
                    if node[2] == i and node[3] == j:
                        ind.append(nameToIndex['x[w][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(-1.0)
                        
                # Customer arc
                for node in data['AC']:
                    if node[0] == i and node[1] == j:
                        ind.append(nameToIndex['x[c][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(1.0)
                    if node[2] == i and node[3] == j:
                        ind.append(nameToIndex['x[c][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(-1.0)
                        
                # Relocation arc
                for node in data['AR']:
                    if node[0] == i and node[1] == j:
                        ind.append(nameToIndex['x[r][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(1.0)
                    if node[2] == i and node[3] == j:
                        ind.append(nameToIndex['x[r][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(-1.0)
                indicesConstr.append(ind)
                coefsConstr.append(co)
                sensesConstr.append('E')
                rhsConstr.append(0)  
        
    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
    


    #######################################
    ####- Relocator flow conservation -####
    #######################################
    
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    
    # Each relocator departs only from one station at time 0
    for q in range(data['Q']):
        ind = []
        co = []
        for i in range(data['S']):
            # Waiting arc
            for node in data['AW']:
                if node[0] == i and node[1] == 0:
                    ind.append(nameToIndex['y[w][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                    co.append(1.0)
                    
            # Customer arc
            for node in data['AT']:
                if node[0] == i and node[1] == 0:
                    ind.append(nameToIndex['y[t][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                    co.append(1.0)
                    
            # Relocation arc
            for node in data['AR']:
                if node[0] == i and node[1] == 0:
                    ind.append(nameToIndex['y[r][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                    co.append(1.0)
                
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(1.0)
    
    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
    
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    # Relocator conservation for all nodes
    for i in range(data['S']):
        for q in range(data['Q']):
            for j in range(1, data['T']):
                ind = []
                co = []
                # Waiting arc
                for node in data['AW']:
                    if node[0] == i and node[1] == j:
                        ind.append(nameToIndex['y[w][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(1.0)
                    if node[2] == i and node[3] == j:
                        ind.append(nameToIndex['y[w][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(-1.0)
                        
                # Customer arc
                for node in data['AT']:
                    if node[0] == i and node[1] == j:
                        ind.append(nameToIndex['y[t][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(1.0)
                    if node[2] == i and node[3] == j:
                        ind.append(nameToIndex['y[t][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(-1.0)
                        
                # Relocation arc
                for node in data['AR']:
                    if node[0] == i and node[1] == j:
                        ind.append(nameToIndex['y[r][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(1.0)
                    if node[2] == i and node[3] == j:
                        ind.append(nameToIndex['y[r][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
                        co.append(-1.0)
                indicesConstr.append(ind)
                coefsConstr.append(co)
                sensesConstr.append('E')
                rhsConstr.append(0)  
        
    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])    


    #######################################
    ##### - Relocator-vehicle match - #####
    #######################################
    
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    # Relocated vehicles smaller than relocating relocators
    for node in data['AR']:
        ind = []
        co = []
        for v in range(data['H']):
            ind.append(nameToIndex['x[r][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
            co.append(1.0)
        for q in range(data['Q']):
            ind.append(nameToIndex['y[r][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
            co.append(-1.0)    
            
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('L')
        rhsConstr.append(0) 
        
    # Relocated vehicles capacity
    for node in data['AR']:
        ind = []
        co = []
        for q in range(data['Q']):
            ind.append(nameToIndex['y[r][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
            co.append(1.0)    
        for v in range(data['H']):
            ind.append(nameToIndex['x[r][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                              + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']'])
            co.append(-b)
            
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('L')
        rhsConstr.append(0) 
        
    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])       


    #######################################
    #### ---- Print and warm start ---- ###
    #######################################

    print('CPLEX model: all constraints added. N constraints: %r. Time: %r\n'\
          %(model.linear_constraints.get_num(), round(time.time()-t_in,2)))


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
        
        results['AC'] = {}
        results['AWH'] = {}
        results['AWQ'] = {}
        results['AT'] = {}
        results['ARH'] = {}
        results['ARQ'] = {}

        ### SAVE RESULTS
        # Vehicles
        for v in range(data['H']):        
            for node in data['AC']:
                results['AC'][str((v,node[0],node[1],node[2],node[3]))] = model.solution.get_values('x[c][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
            for node in data['AW']:
                results['AWH'][str((v,node[0],node[1],node[2],node[3]))] = model.solution.get_values('x[w][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
            for node in data['AR']:
                results['ARH'][str((v,node[0],node[1],node[2],node[3]))] = model.solution.get_values('x[r][' + str(v) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
                    
        # Relocators
        for q in range(data['Q']):        
            for node in data['AT']:
                results['AT'][str((q,node[0],node[1],node[2],node[3]))] = model.solution.get_values('y[t][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
            for node in data['AW']:
                results['AWQ'][str((q,node[0],node[1],node[2],node[3]))] = model.solution.get_values('y[w][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')
            for node in data['AR']:
                results['ARQ'][str((q,node[0],node[1],node[2],node[3]))] = model.solution.get_values('y[r][' + str(q) + ']' + '[' + str(node[0]) + ']' + '[' + str(node[1]) + ']'\
                          + '[' + str(node[2]) + ']' + '[' + str(node[3]) + ']')

            
        results['profits'] = model.solution.get_objective_value()
        return results
    
    except CplexSolverError as e:
        raise Exception('Exception raised')

    return results




if __name__ == '__main__':
    
    # hour = 3
    # time_limit = 3600*hour

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

        

