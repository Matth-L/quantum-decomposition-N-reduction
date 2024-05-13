######################################################################

# Lapu Matthias 

# The goal is to create a controlled gate that will apply the X gate 
# using the ABC decomposition.

######################################################################

import numpy as np
from math import *
import matplotlib.pyplot as plt
from qat.lang.AQASM import *
from qat.qpus import PyLinalg
from qat.lang.AQASM import AbstractGate


def Phase_generator(theta):
    return np.array([[np.exp(1j * theta), 0],
    [0, np.exp(1j * theta)]])


## This function will build the circuit that applies the ABC decomposition
# with the values we chose. It will normally output a controlled X gate.
# Reminder : a controlled gate works like a if then else statement.
def X_controlled_gate(alpha,theta2,theta1,theta0,qbit_to_1=False):

    GlobalPhase = AbstractGate("Phase", [float], arity=1, 
                            matrix_generator=Phase_generator)

    prog = Program()
    qbits = prog.qalloc(2)

    # If we want to test if the control works, 
    # we can change the value of the first qbit before the circuit.
    if qbit_to_1:
        prog.apply(X, qbits[0])

    # Building the circuit

    # Then the C gate
    prog.apply(RZ((theta0-theta2)/2), qbits[1])

    # Then the CNOT
    prog.apply(CNOT,qbits[0], qbits[1])

    #We put B on the first qbit
    prog.apply(RZ((-theta0-theta2)/2), qbits[1])
    prog.apply(RY(-theta1/2), qbits[1])

    # Then the CNOT
    prog.apply(CNOT,qbits[0], qbits[1])

    # We put the Rz rotation on the second qbit
    prog.apply(GlobalPhase(alpha), qbits[0])
    
    # We put A on the first qbit
    prog.apply(RY(theta1/2), qbits[1])
    prog.apply(RZ(theta2), qbits[1])

    # Running the circuit
    circuit = prog.to_circ()
    circuit.display()

    job = circuit.to_job()

    linalgqpu = PyLinalg()

    # printing the result and showing the probabilities
    result = linalgqpu.submit(job)
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
    
# We said previously that : 
# alpha = π/2
# theta2 = 0 
# theta1 = π
# theta0 = π

X_controlled_gate(np.pi/2,0,np.pi,np.pi)

# A controlled gate works like a if then else statement.
# We put |00> in input, if the first qbit is 1, 
# we apply the X gate on the second qbit.
# By default, both qbits are set to 0, so we should have |00> in output.

# The output is indeed |00> with 100% probability.

# If we change the value of the first qbit to 1, we should have |11> in output.

X_controlled_gate(np.pi/2,0,np.pi,np.pi,True)

# The output is indeed |11> with 100% probability.