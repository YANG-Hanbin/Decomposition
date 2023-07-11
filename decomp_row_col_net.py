from gurobipy import Model, GRB, read, LinExpr, disposeDefaultEnv
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import time
import os

def decomp_model(A, sizeA, con, nonzeros, vars, instance, nBlocks, weight_r = 1, weight_c = 5, flag = 1): 
    print("Create a Row-Column-Net Hypergraph.")
    # Create Row-Column-Net Hypergraph
    rNetHg = []
    cNetHg = []
    for i in range(sizeA[0]):
        rNetHg.append(A.getrow(i).nonzero()[1])

    for j in range(sizeA[1]):
        cNetHg.append(A.getcol(j).nonzero()[0])   
                    
    nNodes = len(A.nonzero()[0])
    nHedges = sum(sizeA)
    nHRedges = sizeA[0]
    nHCedges = sizeA[1]
    if flag == 1:
        print("We do weighted partition")
        wrtStr = str(nHedges)+'\t'+str(nNodes)+'\t'+ str(flag)+'\n' # Convert the number of hyperedges nHedges and the number of nodes nNodes into strings, separated by tabs and newlines

        var_ind = {}
        ind_var = {}
        ind = 1

        for i in range(nHRedges):
            wrtStr += str(weight_r)
            wrtStr += '\t'
            for k in range(len(rNetHg[i])):
                var_ind[i, rNetHg[i][k]] = ind 
                ind_var[ind] = (i, rNetHg[i][k])
                wrtStr += str(ind)
                ind += 1
                if k < len(rNetHg[i])-1:
                    wrtStr += '\t'
            wrtStr += '\n'

        for j in range(nHCedges):
            wrtStr += str(weight_c)
            wrtStr += '\t'
            for k in range(len(cNetHg[j])):
                ind = var_ind[cNetHg[j][k], j]
                wrtStr += str(ind)
                if k < len(rNetHg[i])-1:
                    wrtStr += '\t'
            wrtStr += '\n'
    else:
        print("We do un-weighted partition")
        wrtStr = str(nHedges)+'\t'+str(nNodes)+'\n' # Convert the number of hyperedges nHedges and the number of nodes nNodes into strings, separated by tabs and newlines

        var_ind = {}
        ind_var = {}
        ind = 1

        for i in range(nHRedges):
            for k in range(len(rNetHg[i])):
                var_ind[i, rNetHg[i][k]] = ind 
                ind_var[ind] = (i, rNetHg[i][k])
                wrtStr += str(ind)
                ind += 1
                if k < len(rNetHg[i])-1:
                    wrtStr += '\t'
            wrtStr += '\n'

        for j in range(nHCedges):
            for k in range(len(cNetHg[j])):
                ind = var_ind[cNetHg[j][k], j]
                wrtStr += str(ind)
                if k < len(rNetHg[i])-1:
                    wrtStr += '\t'
            wrtStr += '\n'
        
    # print(os.getcwd())
    os.chdir('/Users/aaron/Decomposition/Tests/readMPS')
    f = open('/Users/aaron/Decomposition/Tests/readMPS/HGraphFiles/'+instance+'rcHG','w')
    f.write(wrtStr)
    f.close


    # Use hmetis to decompose the constraint matrix
    os.chdir('/Users/aaron/Decomposition/hmetis-1.5-osx-i686')
    #os.chdir('../../hmetis-1.5-linux/')

    open('../Tests/readMPS/HGraphFiles/'+instance+'rcHG')
    print('shmetis ../Tests/readMPS/HGraphFiles/'+instance+'rcHG '+str(nBlocks)+' 1')
    print("Current directory:", os.getcwd())
    #os.system('/bin/bash shmetis ../Tests/readMPS/HGraphFiles/'+instance+'rcHG '+str(nBlocks)+' 1')
    # Use the HMetis tool to decompose the hypergraph
    if not os.path.exists('HGraphFiles/'+instance+'rcHG.part.'+str(nBlocks)): 
        # If the decomposition file does not exist, call the shmetis command to decompose the hypergraph, and store the result in a file under the HGraphFiles directory
        os.system('./shmetis ../Tests/readMPS/HGraphFiles/'+instance+'rcHG '+str(nBlocks)+' 1')
    else:
        print("Hypergraph exist!!")

    # Read and plot the reordered constraint matrix
    #os.chdir('../')
    os.chdir('../Tests/readMPS/')
    print(os.getcwd())
    f = open('HGraphFiles/'+instance+'rcHG.part.'+str(nBlocks))
    xx = f.readlines() # Read the above decomposed file content into the list 'xx'
    f.close

    # coefgroup: re-group the coeff. corresponding to hypergraph partition
    rowgroup = {}
    colgroup = {}
    coefgroup = {}
    coefToGroup = {}
    for i in range(nBlocks+1):
        coefgroup[i] = [] # each group has an empty set; coefgroup[i] contains all the coef. in block i
        rowgroup[i]  = []
        colgroup[i]  = []


    for i in range(1, len(xx) + 1):
        # xx[i] should be the group number of variable i, i.e., xx[i] = 1,2,...,nBlocks
        coefgroup[int(xx[i - 1].strip('\n'))].append(i) # 'xx' contains the group info from hypergraph partition
        coefToGroup[i] = int(xx[i - 1].strip('\n'))     # variable i belongs to which group

    # re-group variables and constraints according to partition results
    for i in range(sizeA[0]):
        groupind = coefToGroup[var_ind[(i, A.getrow(i).nonzero()[1][0])]]
        if all([coefToGroup[var_ind[(i, A.getrow(i).nonzero()[1][k])]] == groupind for k in range(len(A.getrow(i).nonzero()[1]))]):
            rowgroup[groupind].append(i)
        else:
            # otherwise, it is a linking constraint 
            rowgroup[nBlocks].append(i)

    for j in range(sizeA[1]):
        groupind = coefToGroup[var_ind[(A.getcol(j).nonzero()[0][0], j)]]
        if all([coefToGroup[var_ind[(A.getcol(j).nonzero()[0][k], j)]] == groupind for k in range(len(A.getcol(j).nonzero()[0]))]):
            colgroup[groupind].append(j)
        else:
            # otherwise, it is a linking variable 
            colgroup[nBlocks].append(j)

    # varmap & rowmap are used to re-construct constraints (i.e., sense, variables, RHS) in a right order
    varmap = {}
    ind = 0
    for i in range(nBlocks+1):
        for k in colgroup[i]:
            varmap[k] = ind
            ind += 1

    rowmap = {}
    ind = 0
    for i in range(nBlocks+1):
        for k in rowgroup[i]:
            rowmap[k] = ind
            ind += 1

    # re-construct the new constraint matrix, and plot
    A_coo = A.tocoo(copy=True) # COO(Coordinate Format) is a format for sparse matrix --> (row, col, data); copy = True/False
    A_row = A_coo.row
    A_col = A_coo.col
    A_data = A_coo.data
    A_reord_row = [rowmap[i] for i in A_row]
    A_reord_col = [varmap[i] for i in A_col]
    # A_reord is only for plotting, its corresponding data is incorrect
    A_reord = csr_matrix((A_data,(A_reord_row,A_reord_col))) # create a Compressed Sparse Row format (CSR) sparse matrix
    # plot
    plt.spy(A_reord,markersize=0.5)
    plt.show()
    os.chdir('../../')

    # re-construct the new constraint matrix
    coef_group = [x.strip() for x in xx]

    counter = {}

    for element in coef_group:
        counter[element] = counter.get(element, 0) + 1

    element_counts = list(counter.items())
    element_counts.sort(key=lambda x: x[1])

    lowest_count = element_counts[0][1]
    highest_count = element_counts[-1][1]
    
    # add slack variable
    add_slack = {}
    num_linking_cons = 0
    for i in range(nBlocks):
        for row in rowgroup[i]:
            add_slack[row] = True

    for row in rowgroup[nBlocks]:
        add_slack[row] = False
        num_linking_cons += 1

    # copy variables 
    add_copy_var = {}
    num_copy_var = 0
    for i in range(nBlocks):
        for col in colgroup[i]:
            for row in A.getcol(col).nonzero()[0]:
                add_copy_var[(row, col)] = False
    for col in colgroup[nBlocks]:
        for row in A.getcol(col).nonzero()[0]:
            add_copy_var[(row, col)] = True
            num_copy_var += 1

    return add_slack, add_copy_var, num_linking_cons, num_copy_var, lowest_count, highest_count



def solve_decomp_with_slack(A, vars, RHS, SENSE, add_slack, add_copy_var):
    new_model = Model()
    new_model.modelSense = GRB.MINIMIZE
    new_model.update()

    # add variables to the model
    X_vars = {}
    for i in range(A.shape[1]):
        X_vars[i] = new_model.addVar(lb = vars[i].lb, ub = vars[i].ub, vtype = vars[i].vtype,
                                                name = "X" + str(i))
    new_model.update()

    slack_vars = []
    Y_vars = {}
    Diff = {}
    objExpr = LinExpr()

    for i in range(A.shape[0]):
        ConsExpr = LinExpr()
        for j in A.getrow(i).nonzero()[1]:
            if add_copy_var[i,j]:
                Y_vars[i,j] = new_model.addVar(lb = vars[j].lb, ub = vars[j].ub, vtype = vars[j].vtype,
                                                name = "Y" + str(i) + "_" + str(j))
                Diff[i,j] = new_model.addVar(lb = 0.0, vtype=GRB.CONTINUOUS)
                new_model.addConstr(Diff[i,j] >= X_vars[j] - Y_vars[i,j])
                new_model.addConstr(Diff[i,j] >= - X_vars[j] + Y_vars[i,j])
                objExpr += Diff[i,j]
                ConsExpr += A[i,j]*Y_vars[i,j]
            else: 
                ConsExpr += A[i,j]*X_vars[j]
        if add_slack[i] and SENSE[i] == "<":
            cur_slack = new_model.addVar(lb = 0.0, vtype = GRB.CONTINUOUS,
                                                name = "S" + str(i))
            slack_vars.append(cur_slack)
            ConsExpr -= cur_slack
        elif add_slack[i] and SENSE[i] == ">":
            cur_slack = new_model.addVar(lb = 0.0, vtype = GRB.CONTINUOUS,
                                                name = "S" + str(i))
            slack_vars.append(cur_slack)
            ConsExpr += cur_slack
        elif add_slack[i] and SENSE[i] == "=":
            cur_slack1 = new_model.addVar(lb = 0.0, vtype = GRB.CONTINUOUS,
                                                name = "S" + str(i) + '_1')
            cur_slack2 = new_model.addVar(lb = 0.0, vtype = GRB.CONTINUOUS,
                                                name = "S" + str(i) + '_2')
            slack_vars.append(cur_slack1)
            slack_vars.append(cur_slack2)
            ConsExpr += cur_slack1 - cur_slack2
        new_model.addConstr(lhs = ConsExpr, sense = SENSE[i], rhs = RHS[i], name = 'Constr'+str(i))  

    new_model.update()
    
    for i in range(len(slack_vars)):
        objExpr += slack_vars[i]
    
    new_model.setObjective(objExpr, GRB.MINIMIZE)
    
    new_model.setParam(GRB.Param.Seed, 77)
    new_model.setParam('TimeLimit', 10*60)

    start_time = time.time()
    new_model.optimize()
    end_time = time.time()
    feasibility_time = end_time - start_time

    status = new_model.status

    print(new_model.getObjective().getValue())
    
    return feasibility_time, status



