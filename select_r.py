#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyqpanda import *
import numpy as np


# ## Init Quantum Environment

# In[2]:


class InitQMachine:
    def __init__(self, qubitsCount, cbitsCount, machineType = QMachineType.CPU):
        self.machine = init_quantum_machine(machineType)
        
        self.qubits = self.machine.qAlloc_many(qubitsCount)
        self.cbits = self.machine.cAlloc_many(cbitsCount)
        
        print(f'Init Quantum Machine with qubits:[{qubitsCount}] / cbits:[{cbitsCount}] Successfully')
    
    def __del__(self):
        destroy_quantum_machine(self.machine)


# In[3]:


ctx = InitQMachine(4, 4)

machine = ctx.machine
qubits = ctx.qubits
cbits = ctx.cbits


# # 0. Tool Functions

# In[4]:


Dag = lambda matrix: matrix.conj().T


# In[5]:


normalize = lambda v: v / np.linalg.norm(v)


# In[6]:


def is_hermitian(matrix):
    return np.allclose(matrix, Dag(matrix))


# In[7]:


def ed(m, n):
     return np.sqrt(np.sum((m - n) ** 2))


# # 1. Solve using numpy

# ## 1.0 solve original problem

# In[8]:


A = np.array([
    [1, 1],
    [2 ** 0.5 / 2, -(2 ** 0.5) / 2]
])

b = np.array([
    [0.5], [-(2 ** 0.5) / 2]
])


# In[9]:


# solve: Ax = b
x = np.linalg.solve(A, b)
x


# ## 1.1 test A†Ax = A†b

# In[10]:


Dag(A)


# In[11]:


A_ = Dag(A) @ A # make A hermitian
b_ = Dag(A) @ b


# In[12]:


x = np.linalg.solve(A_, b_)
x


# ## 1.2 eigenvalue of A

# In[13]:


eigenvalue, _ = np.linalg.eig(A_)
eigenvalue


# # 2. HHL

# ## 2.1 make A hermitian

# In[14]:


A_ = Dag(A) @ A # make A hermitian


# In[15]:


# test is_hermitian
print(is_hermitian(A)) # false
print(is_hermitian(A_)) # true


# ## 2.2 HHL algorithm subroutines

# ### 2.2.1 encode b

# In[16]:


QOperator(X(qubits[0])).get_matrix()


# In[17]:


def encode(b):
    circuit = create_empty_circuit()
#     circuit << amplitude_encode(qubits[3], b)
    circuit << X(qubits[3])
    
    return circuit


# In[18]:


# https://arxiv.org/pdf/1110.2232.pdf


# ### 2.2.2 phase estimation

# In[19]:


def phase_estimation(A):
    circuit = create_empty_circuit()
    
    circuit << H(qubits[1]) << H(qubits[2]) << BARRIER(qubits[1:3])
    
#     circuit << QOracle(qubits[3], expMat(1j, A, np.pi / 2)).control(qubits[2]) # C-U^1
#     circuit << QOracle(qubits[3], expMat(1j, A, np.pi)).control(qubits[1]) # C-U^2
    circuit << CU(-np.pi / 4, -3 * np.pi / 2, -3 * np.pi / 2, 3 * np.pi / 2, qubits[2], qubits[3])
    circuit << CU(-3 * np.pi/2, -3 * np.pi, -3 * np.pi, -2 * np.pi, qubits[1], qubits[3])
    circuit << BARRIER(qubits[1:3])
    
    # inverse QFT
    circuit << SWAP(qubits[1], qubits[2])
    circuit << H(qubits[2])
    circuit << S(qubits[2]).dagger().control(qubits[1])
    circuit << H(qubits[1])
    circuit << SWAP(qubits[1], qubits[2])

    return circuit


# ### 2.2.4 controlled rotations

# In[20]:


def rotation(r):
    circuit = create_empty_circuit()
    
    circuit << RY(qubits[0], 2*np.pi/(2**r)).control(qubits[1])
    circuit << RY(qubits[0], np.pi/(2**r)).control(qubits[2])
    
    return circuit


# ### 2.2.5 uncompute

# In[21]:


def uncompute(A):
    circuit = create_empty_circuit()
    
    # QFT
    circuit << SWAP(qubits[1], qubits[2])
    circuit << H(qubits[1])
    circuit << S(qubits[2]).control(qubits[1])
    circuit << H(qubits[2])
    circuit << SWAP(qubits[1], qubits[2])
    circuit << BARRIER(qubits[1:3])

#     circuit << QOracle(qubits[3], expMat(-1j, A, np.pi)).control(qubits[1])
#     circuit << QOracle(qubits[3], expMat(-1j, A, np.pi / 2)).control(qubits[2])
    circuit << CU(-3 * np.pi/2, -3 * np.pi, -3 * np.pi, -2 * np.pi, qubits[1], qubits[3])
    circuit << CU(np.pi/4, -3*np.pi/2, -np.pi/2, -np.pi/2, qubits[2], qubits[3]) << BARRIER(qubits[1:3])
    
    circuit << H(qubits[1]) << H(qubits[2])
    
    return circuit


# ## 2.3 full HHL algorithm 

# In[22]:


def HHL(A, b, r, flag=False):
    prog = create_empty_qprog()
    
    # Step 0. check input
    if not is_hermitian(A):
        b = (Dag(A) @ b).round(4)
        A = (Dag(A) @ A).round(4) # make A hermitian
    
    normed_b = (b / np.linalg.norm(b)).round(4)
    
    # Step 1. state preparation
    prog << encode(normed_b)
    
    # Step 2. phase estimation
    prog << phase_estimation(A)
    
    # Step 3. rotation
    prog << rotation(r)
    
    # Step 4. uncompute
    prog << uncompute(A)
    
    # Step 5. measure ancilla qubit
    prog << Measure(qubits[0], cbits[0])
    if flag:
        results = [0, 0]
        for i in range(10000):
            result = prob_run_list(prog, qubits[0], -1)
            results[0] += result[0]
            results[1] += result[1]

        results[0] /= 10000
        results[1] /= 10000
        
        return results
    
    result = directly_run(prog)
    
    if not result['c0']:
        return HHL(A, b, r)
    
    # Step 6. get results
    prog << Measure(qubits[3], cbits[3])
    
    qstate = get_qstate()
    normed_x = np.real(np.array([qstate[1], qstate[9]])) # 0001 1001
        
    # Step 7. recover x
    N = len(normed_b)
    ratio = 0.0
    for i in range(N):
        if not abs(normed_b[i]) < 1e-8:
            ratio = normed_b[i][0] / np.sum([ normed_x[j] * A[i][j] for j in range(N) ])
            break
    
    # normed_x = x / ||x|| => x = normed_x * ||x||
    x_ = (normed_x * ratio)
    distance = ed(x_, x.flatten())
    
    return distance, x_


# ## 2.4 select proper r for rotation

# In[23]:


r = [1, 2, 3, 4, 5, 6, 7, 8]


# In[24]:


p = [ HHL(A, b, i, True)[1] for i in r]


# In[25]:


distances = [ HHL(A, b, i)[0] for i in r ]


# In[26]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


plt.figure(figsize=(15, 4))
plt.subplot(1,2,1)
plt.scatter(r, p)
plt.xlabel("r", fontsize=15)
plt.ylabel("Probability", fontsize=15)

plt.subplot(1,2,2)
plt.scatter(r, distances, color="#FF6600")
plt.xlabel("r", fontsize=15)
plt.ylabel("Distance", fontsize=15)

plt.show()


# In[ ]:




