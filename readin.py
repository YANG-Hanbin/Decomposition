from gurobipy import Model, GRB, read, LinExpr, disposeDefaultEnv
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import time
import os
import pandas as pd


def read_model(instance):
    m = read('instances/'+instance+'.mps.gz')
    m_new = read('instances/'+instance+'.mps.gz')
    m_new1 = read('instances/'+instance+'.mps.gz')
    m_new2 = read('instances/'+instance+'.mps.gz')
    return m, m_new, m_new1, m_new2

def get_info(m):
    A = m.getA()
    #print(A)
    # plot sparsity of A
    #plt.spy(A,markersize=0.5)
    #plt.show()
    x = m.getVars()
    for i in range(len(x)):
        x[i].VarName = 'x_'+str(i)
    con = m.getConstrs()
    #print(len(x),len(con))
    sizeA = A.get_shape()
    #print(sizeA)
    A.eliminate_zeros()
    nonzeros = A.nonzero()
    
    # get rhs of orgininal problem
    RHS = m.getAttr('RHS')
    SENSE = m.getAttr('Sense')
    # Get the variables in the model
    vars = m.getVars()
    
    return A, x, con, sizeA, nonzeros, RHS, SENSE, vars