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


    # The Sleathor Weinfurter reduction is the following :
    # - apply the V gate on the 2nd and 3rd qubits
    # - apply a CNOT gate on the 1st and 2nd qubits
    # - apply the V dagger gate on the 2nd and 3rd qubits
    # - apply a CNOT gate on the 1st and 2nd qubits
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

for state in [[0,0,0], [0,0,1], [0,1,0], [0,1,1], 
              [1,0,0], [1,0,1], [1,1,0], [1,1,1]]:
    Sleathor_Weinfurter_2qubits_X(X_squared_gate, X_squared_gate_dagger, state)

