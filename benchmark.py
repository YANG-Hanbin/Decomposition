from gurobipy import Model, GRB, read, LinExpr, disposeDefaultEnv
import gurobipy as gp
import time


def solve_obj_0(m_new1):
    # set the obj to 0 for the original problem to get a feasible solution

    # Set a new objective function
    m_new1.setObjective(0, GRB.MINIMIZE)
    
    m_new1.setParam(GRB.Param.Seed, 77)

    start_time = time.time()

    # Optimize the modified model
    m_new1.optimize()

    end_time = time.time()

    obj_to_0_time = end_time - start_time
    
    status = m_new1.status
    
    print(m_new1.getObjective().getValue())
    
    return obj_to_0_time, status
    
    
def solve_all_add_slack(A, vars, SENSE, RHS):
    # all add slack variable
    new_model_all = Model()
    new_model_all.modelSense = GRB.MINIMIZE
    new_model_all.update()

    X_vars_all = [0 for i in range(A.shape[1])]
    for i in range(A.shape[1]):
        curVar = new_model_all.addVar(lb = vars[i].lb, ub = vars[i].ub, vtype = vars[i].vtype,
                                               name = "X" + str(i))
        X_vars_all[i] = curVar

    new_model_all.update()
        

    slack_vars_all = []
    for i in range(A.shape[0]):
        ConsExpr = LinExpr()

        for j in A.getrow(i).nonzero()[1]:  # .nonzero -> [0]: first indices, [1]: second indices, [2]: corresponding values
            ConsExpr += A[i,j]*X_vars_all[j]
        if SENSE[i] == "<":
            cur_slack = new_model_all.addVar(lb = 0.0, vtype = GRB.CONTINUOUS,
                                               name = "S" + str(i))
            slack_vars_all.append(cur_slack)
            ConsExpr -= cur_slack
        elif SENSE[i] == ">":
            cur_slack = new_model_all.addVar(lb = 0.0, vtype = GRB.CONTINUOUS,
                                               name = "S" + str(i))
            slack_vars_all.append(cur_slack)
            ConsExpr += cur_slack
        elif SENSE[i] == "=":
            cur_slack1 = new_model_all.addVar(lb = 0.0, vtype = GRB.CONTINUOUS,
                                               name = "S" + str(i) + '_1')
            cur_slack2 = new_model_all.addVar(lb = 0.0, vtype = GRB.CONTINUOUS,
                                               name = "S" + str(i) + '_2')
            slack_vars_all.append(cur_slack1)
            slack_vars_all.append(cur_slack2)
            ConsExpr += cur_slack1 - cur_slack2
        
        new_model_all.addConstr(lhs = ConsExpr, sense = SENSE[i], rhs = RHS[i], name = 'Constr'+str(i))  

    new_model_all.update()

    objExpr = LinExpr()

    for i in range(len(slack_vars_all)):
        objExpr += slack_vars_all[i]
    
    new_model_all.setObjective(objExpr, GRB.MINIMIZE)
    
    new_model_all.setParam(GRB.Param.Seed, 77)

    start_time = time.time()
    new_model_all.optimize()
    end_time = time.time()

    all_add_slack_time = end_time - start_time
    
    status = new_model_all.status
    
    print(new_model_all.getObjective().getValue())
    
    return all_add_slack_time, status
