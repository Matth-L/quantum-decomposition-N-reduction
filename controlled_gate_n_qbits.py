######################################################################

# Lapu Matthias 

# The goal is to create a controled gate on n qbits, using NOT that
# controls multiple qbits.
######################################################################

from math import *
import numpy as np
from qat.qpus import PyLinalg
from qat.lang.AQASM import AbstractGate
from qat.lang.AQASM import *
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


# We need to create the X squared gate, because we apply it to 2 qubits
# we need it to be a 4x4 matrix
def X_squared_gate():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0,np.sqrt(2)/2, -1j*np.sqrt(2)/2],
                     [0, 0, -1j*np.sqrt(2)/2, np.sqrt(2)/2]])

# We will also need the dagger of the X squared gate
# it's the adjoint of the matrix
def X_squared_gate_dagger():
    return X_squared_gate().conj().T

# To build the controlled gate using 3qbits, we need the fourth root 
# of the X gate
def X_fourth_gate():
    return sqrtm(X_squared_gate())

def X_fourth_gate_dagger():
    return X_fourth_gate().conj().T

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


## Build the Sleathor Weinfurter reduction with 2 qbits :w!
def Sleathor_Weinfurter_2qubits_X(prog,mat,mat_dagger,starting_qbit:int):
    
    V = AbstractGate("V", [], arity=2, matrix_generator=mat)
    VD = AbstractGate("VD", [], arity=2, matrix_generator=mat_dagger)

    final_qbit = starting_qbit+3 # we must hop one of the qbit   
    # Building the circuit
    prog.apply(V(), qbits[starting_qbit+1], qbits[final_qbit])
    prog.apply(CNOT, qbits[starting_qbit], qbits[starting_qbit+1])
    prog.apply(VD(), qbits[starting_qbit+1], qbits[final_qbit])
    prog.apply(CNOT, qbits[starting_qbit], qbits[starting_qbit+1])
    prog.apply(V(), qbits[starting_qbit], qbits[final_qbit])


## Build the Sleathor Weinfurter reduction with 3 qbits
# explained in Sleahtor_Weinfurter_reduction.py
def Sleathor_Weinfurter_3qubits_X(prog,mat,mat_dagger,
                                  starting_qbit:int,):

    # the starting qbit is (from the top) where the firts qbit available 
    # for the scope of the reduction is located

    # Then, we create the V and V dagger gates using the function 
    # passed as an argument
    V = AbstractGate("V", [], arity=2, matrix_generator=mat)
    VD = AbstractGate("VD", [], arity=2, matrix_generator=mat_dagger)

    final_qbit = starting_qbit+4    # Building the circuit

    # Building the circuit
    prog.apply(V(),qbits[starting_qbit],qbits[final_qbit])
    prog.apply(CNOT, qbits[starting_qbit], qbits[starting_qbit+1])
    prog.apply(VD(), qbits[starting_qbit+1], qbits[final_qbit])
    prog.apply(CNOT, qbits[starting_qbit], qbits[starting_qbit+1])
    prog.apply(V(), qbits[starting_qbit+1], qbits[final_qbit])
    prog.apply(CNOT, qbits[starting_qbit+1], qbits[starting_qbit+2])
    prog.apply(VD(), qbits[starting_qbit+2], qbits[final_qbit])
    prog.apply(CNOT, qbits[starting_qbit], qbits[starting_qbit+2])
    prog.apply(V(), qbits[starting_qbit+2], qbits[final_qbit])
    prog.apply(CNOT, qbits[starting_qbit+1], qbits[starting_qbit+2])
    prog.apply(VD(), qbits[starting_qbit+2], qbits[final_qbit])
    prog.apply(CNOT, qbits[starting_qbit], qbits[starting_qbit+2])
    prog.apply(V(), qbits[starting_qbit+2], qbits[final_qbit])




# To create a controlled gate on m qbits, 
# we need a recursive CNOT to apply to m-2 qbits located at the m-1 qbit
# (so just 1 before the last qbit)
# We also need the V controlled gate on m-2 qbits located at the m qbit

def controlled_gate_n_qbits_X(m,mat,squared_mat,mat_dagger,squared_mat_dagger):
    prog = Program()
    qbits = prog.qalloc(m)

    # We need to create the controlled CNOT gate on m-2 qbits

    CCNOT = AbstractGate("CCNOT", [], arity=(m-1), 
                         matrix_generator=recursive_CNOT(m-2))

    # We also need to create the V gate and the V dagger gate
    V = AbstractGate("V", [], arity=2, matrix_generator=mat)
    VD = AbstractGate("VD", [], arity=2, matrix_generator=mat_dagger)

    # Then we apply the V gate on the m-2 and m-1 qbits

    prog.apply(V(), qbits[m-2], qbits[m-1])

    # the list of qbits that the cnot gate will control
    qbits_to_control = [qbits[i] for i in range(m-2)]

    # Then we apply the CNOT gate on the m-2 qbits to control the m-3 qbit
    # The syntax error comes from here, I did not find a way to fix it
    # The documentation only talks about gate with a clear number of qbits
    # I did not find any example of a gate that controls variable qbits
    prog.apply(CCNOT,qbits_to_control, qbits[m-1])
    prog.apply(VD(), qbits[m-2], qbits[m-1])
    prog.apply(CCNOT,qbits_to_control, qbits[m-1])
    
    # The only thing we need is to create the V controlled gate on m-2 qbits
    # It can be seen as a U gate, in this case the V and VD gate of this one 
    # would be the fourth root of the X gate

    if m == 4:
        Sleathor_Weinfurter_2qubits_X(prog,squared_mat,squared_mat_dagger,m-1)
    elif m == 5:
        Sleathor_Weinfurter_3qubits_X(prog,squared_mat,squared_mat_dagger,m-2)
    else:
        controlled_gate_n_qbits_X(m-1,mat,squared_mat,
                                  mat_dagger,squared_mat_dagger)
    
     # Running the circuit
    circuit = prog.to_circ()
    circuit.display()

    mypylinalgqpu = get_default_qpu()

    job = circuit.to_job()
    result = mypylinalgqpu.submit(job)

    # Plotting the results and the percentage of each state :w!
    l = len(result)
    states = ['']*l
    probabilities= [0]*l

    i=0
    for sample in result:
        print("State",sample.state,"with amplitude",
                sample.amplitude,"and probability",
                round(sample.probability*100,2),"%")
        states[i] = str(sample.state)
        probabilities[i] = round(sample.probability*100,2)
        i = i+1
    plt.bar(states, probabilities, color='skyblue')
    plt.xlabel('States')
    plt.ylabel('Probabilities')
    plt.show()

controlled_gate_n_qbits_X(5,X_squared_gate,X_fourth_gate,
                          X_squared_gate_dagger,X_fourth_gate_dagger)