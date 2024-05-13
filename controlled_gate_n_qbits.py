######################################################################

# Lapu Matthias 

# The goal is to create a controled gate on n qbits, using NOT that
# controls multiple qbits.
######################################################################

from qat.lang.AQASM import *
from qat.qpus import PyLinalg
from qat.lang.AQASM import AbstractGate
import matplotlib.pyplot as plt
import numpy as np
from math import *
import cmath

## This function will create the CNOT matrix
# Reminder : the X gate matrix is : [[0,1],[1,0]]
# The identity matrix is : [[1,0],[0,1]]
# The CX matrix can be written as a block matrix like this :
# [[I, 0], [0, X]]
def CNOT_matrix():
    # we need to create 2 matrix of size (n*n), the identity matrix
    # and the X gate matrix
    # Identity matrix
    I = np.identity(2)
    Zero = np.zeros((2,2))
    X = np.flip(np.identity(2),0)

    # Creating the CNOT matrix
    CNOT = np.block([[I, Zero],
                    [Zero, X]])
    return CNOT

## The toffoli gate, which is a controlled not gate on 3 qbits can be 
# written as a block matrix like this :
# [[I, 0], [0, CX]] with CX the CNOT matrix written above
# Thus, we can write a recursive function that will create a CNOT gate
# on n qbits
# n=3 should return the Toffoli gate
def recursive_CNOT(n):
    if n == 2:
        return CNOT_matrix()
    else:
        return np.block([[np.identity(2**(n-1)), np.zeros((2**(n-1),2**(n-1)))],
                        [np.zeros((2**(n-1),2**(n-1))), recursive_CNOT(n-1)]])

