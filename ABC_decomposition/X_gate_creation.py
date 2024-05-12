from qat.lang.AQASM import *
from qat.qpus import PyLinalg
from qat.lang.AQASM import AbstractGate
import matplotlib.pyplot as plt
import numpy as np
from math import *
import cmath

# We had 12 values in to choose from : 
# Let's take : 
# alpha = π/2
# theta2 = 0 
# theta1 = π
# theta0 = π

def Phase_generator(theta):
    return np.array([[np.exp(1j * theta), 0],
    [0, np.exp(1j * theta)]])

GlobalPhase = AbstractGate("Phase", [float], arity=1, 
                           matrix_generator=Phase_generator)

## This fonction will build the circuit that applies the ABC decomposition
# with the values we chose. It will normally output the X gate.
# which mean that with 1 qbit set to 0 we should have 1 qbit set to 1 with 100%.
def build_ABC_decomposition_circuit(alpha, theta2, theta1, theta0):

    prog = Program()
    qbits = prog.qalloc(1)


    # Reminder of the formula : 
    # U = exp(i*alpha) * Rz(theta2) * Ry(theta1) * Rz(theta0)
    # We must apply it in the reverse order.

    # We apply the Rz(theta2) gate
    prog.apply(RZ(theta0), qbits[0])
    # Then the Ry(theta1) gate
    prog.apply(RY(theta1), qbits[0])
    # Finally the Rz(theta0) gate
    prog.apply(RZ(theta2), qbits[0])

    # We apply the global phase that corresponds to exp(i*alpha)
    prog.apply(GlobalPhase(alpha), qbits[0])

    # Running the circuit
    circuit = prog.to_circ()
    circuit.display()

    job = circuit.to_job()

    linalgqpu = PyLinalg()

    # printing the result
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

build_ABC_decomposition_circuit(np.pi/2, 0, np.pi, np.pi)

# As we can see, with a qbit of 0, we have a qbit of 1 with 100% probability.
# Thus we have successfully implemented the X gate with the ABC decomposition.