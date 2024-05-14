######################################################################

# Lapu Matthias 

# The goal is to create a circuit that will apply the X gate 
# with a squared gate using the Sleathor Weinfurter reduction

######################################################################

import cmath
from math import *
import numpy as np
import matplotlib.pyplot as plt
from qat.lang.AQASM import *
from qat.qpus import get_default_qpu
from itertools import product
from scipy.linalg import sqrtm

# Must create all the possible states for 2 bits
# like creating_2n_states(2) will return [[0,0],[0,1],[1,0],[1,1]]
def creating_2n_states(n):
    return list(product([0, 1], repeat=n))


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

# We will create a generic function that will apply the 
# Sleathor Weinfurter reduction, thus we need V and V dagger gates
def Sleathor_Weinfurter_2qubits_X(mat,mat_dagger,state):
    
    # Starting the program and allocating 3 qubits
    prog = Program()
    qbits = prog.qalloc(3)

    # Creating the initial state 
    for i in range(len(state)):
        if state[i] == 1:
            prog.apply(X, qbits[i])

    # Then, we create the V and V dagger gates using the function 
    # passed as an argument
    V = AbstractGate("V", [], arity=2, matrix_generator=mat)
    VD = AbstractGate("VD", [], arity=2, matrix_generator=mat_dagger)


    # The Sleathor Weinfurter reduction with 2 qbits is the following :
    # - apply the V gate on the 2nd and 3rd qubits
    # - apply a CNOT gate. The X is on the 2nd qbit, the target is the 1st
    # - apply the V dagger gate on the 2nd and 3rd qubits
    # - apply a CNOT gate. The X is on the 2nd qbit, the target is the 1st
    # - apply the V gate on the 1st and 3rd qubits

    # Building the circuit
    prog.apply(V(), qbits[1], qbits[2])
    prog.apply(CNOT, qbits[0], qbits[1])
    prog.apply(VD(), qbits[1], qbits[2])
    prog.apply(CNOT, qbits[0], qbits[1])
    prog.apply(V(), qbits[0], qbits[2])

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


def Sleathor_Weinfurter_3qubits_X(mat,mat_dagger,state):
    
    # Starting the program and allocating 3 qubits
    prog = Program()
    qbits = prog.qalloc(4)

    # Creating the initial state 
    for i in range(len(state)):
        if state[i] == 1:
            prog.apply(X, qbits[i])

    # Then, we create the V and V dagger gates using the function 
    # passed as an argument
    V = AbstractGate("V", [], arity=2, matrix_generator=mat)
    VD = AbstractGate("VD", [], arity=2, matrix_generator=mat_dagger)


    # The Sleathor Weinfurter reduction with 3 qbits is the following :
    # - apply the V gate on the 1st qbit
    # - apply a CNOT gate. The X is on the 2nd qbit, the target is the 1st
    # - apply a VD gate on the 2nd qbit
    # - apply a CNOT gate. The X is on the 2nd qbit, the target is the 1st
    # - apply the V gate on the 2nd qbit
    # - apply a CNOT gate. The X is on the 3rd qbit, the target is the 2nd
    # - apply a VD gate on the 3rd qbit
    # - apply a CNOT gate. The X is on the 3rd qbit, the target is the 1st
    # - apply the V gate on the 3nd qbit
    # - apply a CNOT gate. The X is on the 3rd qbit, the target is the 2nd
    # - apply a VD gate on the 3rd qbit
    # - apply a CNOT gate. The X is on the 3rd qbit, the target is the 1st
    # - apply the V gate on the 3nd qbit

    # Building the circuit
    prog.apply(V(),qbits[0],qbits[3])
    prog.apply(CNOT, qbits[0], qbits[1])
    prog.apply(VD(), qbits[1], qbits[3])
    prog.apply(CNOT, qbits[0], qbits[1])
    prog.apply(V(), qbits[1], qbits[3])
    prog.apply(CNOT, qbits[1], qbits[2])
    prog.apply(VD(), qbits[2], qbits[3])
    prog.apply(CNOT, qbits[0], qbits[2])
    prog.apply(V(), qbits[2], qbits[3])
    prog.apply(CNOT, qbits[1], qbits[2])
    prog.apply(VD(), qbits[2], qbits[3])
    prog.apply(CNOT, qbits[0], qbits[2])
    prog.apply(V(), qbits[2], qbits[3])


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


## To test if the X squared gate works, 
# if the first qbit is one, the second qbit should switch
# otherwise, the second qbit should stay the same
def testing_X_sq2_sq4(state,th_gate):

    # The test is only for the squared gate and the fourth gate
    if th_gate != 2 and th_gate != 4:
        print("The th_gate must be 2 or 4")
        return
    
    prog = Program()
    qbits = prog.qalloc(2)

    # Creating the gate depending on the th_gate
    if th_gate == 2:
        V = AbstractGate("V", [], arity=2, matrix_generator=X_squared_gate)
    else:
        V = AbstractGate("V", [], arity=2, matrix_generator=X_fourth_gate)


    # Creating the initial state 
    for i in range(len(state)):
        if state[i] == 1:
            prog.apply(X, qbits[i])

    # Applying the gate th_gate times should always return the X gate
    for i in range(th_gate):
        prog.apply(V(), qbits[0], qbits[1])
 
    # Running the circuit
    circuit = prog.to_circ()
    # circuit.display()

    mypylinalgqpu = get_default_qpu()

    job = circuit.to_job()
    result = mypylinalgqpu.submit(job)

    # printing the percentage of each state
    for sample in result:
        print("State",sample.state,", probability",
                round(sample.probability*100,2),"%")


def main():

    # Testing the X squared gate
    for state in creating_2n_states(2):
        print("Testing the X squared gate with the state : ",state)
        testing_X_sq2_sq4(state,2)
        # looks ok
    
    print("-------------------------------------------------")

    # Testing the X fourth gate
    for state in creating_2n_states(2):
        print("Testing the X fourth gate with the state : ",state)
        testing_X_sq2_sq4(state,4)
        # looks ok
    print("-------------------------------------------------")
    
    # Testing the Sleathor Weinfurter reduction with 2 qubits
    for state in creating_2n_states(2):
        print("Testing the Sleathor Weinfurter reduction with the state : ",state)
        Sleathor_Weinfurter_2qubits_X(X_squared_gate, X_squared_gate_dagger, state)

    print("-------------------------------------------------")

    # Testing the Sleathor Weinfurter reduction with 3 qubits
    for state in creating_2n_states(3):
        print("Testing the Sleathor Weinfurter reduction with the state : ",state)
        Sleathor_Weinfurter_3qubits_X(X_fourth_gate, X_fourth_gate_dagger, state)

main()