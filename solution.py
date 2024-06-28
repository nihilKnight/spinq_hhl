#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyqpanda import *
import numpy as np


# In[2]:


def draw(prog, filename=''):
    dir_path = './images/'
    if filename != '':
        draw_qprog(prog, 'pic', filename=f'{dir_path}{filename}')


# ## Init Quantum Environment

# In[3]:


class InitQMachine:
    def __init__(self, qubitsCount, cbitsCount, machineType = QMachineType.CPU):
        self.machine = init_quantum_machine(machineType)
        
        self.qubits = self.machine.qAlloc_many(qubitsCount)
        self.cbits = self.machine.cAlloc_many(cbitsCount)
        
        print(f'Init Quantum Machine with qubits:[{qubitsCount}] / cbits:[{cbitsCount}] Successfully')
    
    def __del__(self):
        destroy_quantum_machine(self.machine)


# In[4]:


ctx = InitQMachine(4, 4)

machine = ctx.machine
qubits = ctx.qubits
cbits = ctx.cbits


# # 0. Tool Functions

# In[7]:


Dag = lambda matrix: matrix.conj().T


# In[9]:


normalize = lambda v: v / np.linalg.norm(v)


# In[28]:


def is_hermitian(matrix):
    return np.allclose(matrix, Dag(matrix))


# # 1. Solve using numpy

# ## 1.0 solve original problem

# In[19]:


A = np.array([
    [1, 1],
    [2 ** 0.5 / 2, -(2 ** 0.5) / 2]
])

b = np.array([
    [0.5], [-(2 ** 0.5) / 2]
])


# In[20]:


# solve: Ax = b
x = np.linalg.solve(A, b)
x


# ## 1.1 test A†Ax = A†b

# In[21]:


Dag(A)


# In[22]:


A_ = Dag(A) @ A # make A hermitian
b_ = Dag(A) @ b


# In[23]:


x = np.linalg.solve(A_, b_)
x


# ## 1.2 eigenvalue of A

# In[27]:


eigenvalue, _ = np.linalg.eig(A_)
eigenvalue


# # 2. HHL

# ## 2.1 make A hermitian

# In[30]:


A_ = Dag(A) @ A # make A hermitian


# In[31]:


# test is_hermitian
print(is_hermitian(A)) # false
print(is_hermitian(A_)) # true


# ## 2.2 HHL algorithm subroutines

# ### 2.2.1 encode b

# In[35]:


QOperator(X(qubits[0])).get_matrix()


# In[33]:


def encode(b):
    circuit = create_empty_circuit()
#     circuit << amplitude_encode(qubits[3], b)
    circuit << X(qubits[3])
    
    return circuit


# In[34]:


draw(encode(b_), 'encode_x')


# In[36]:


# https://arxiv.org/pdf/1110.2232.pdf


# ### 2.2.2 phase estimation

# In[38]:


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


# In[39]:


draw(phase_estimation(A_), 'phase_estimation')


# ### 2.2.4 controlled rotations

# In[40]:


def rotation():
    circuit = create_empty_circuit()
    
    circuit << RY(qubits[0], np.pi / 32).control(qubits[2])
    circuit << RY(qubits[0], np.pi / 16).control(qubits[1])
    
    return circuit


# In[41]:


draw(rotation(), 'rotation')


# ### 2.2.5 uncompute

# In[43]:


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


# In[44]:


draw(uncompute(A_), 'uncompute')


# ## 2.3 full HHL algorithm 

# In[45]:


def HHL(A, b):
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
    prog << rotation()
    
    # Step 4. uncompute
    prog << uncompute(A)
    
    prog << BARRIER(qubits)
    
    # Step 5. measure ancilla qubit
    prog << Measure(qubits[0], cbits[0])
    
    result = directly_run(prog)
    if not result['c0']:
#         print('attempting...')
        return HHL(A, b)
    
    # Step 6. get results
    qstate = get_qstate()
    normed_x = np.real(np.array([qstate[1], qstate[9]])) # 0001 1001
    
    # Step 7. recover x
    N = len(normed_b)
    ratio = 0.0
    for i in range(N):
        if not abs(normed_b[i]) < 1e-8:
            ratio = normed_b[i][0] / np.sum([ normed_x[j] * A[i][j] for j in range(N) ])
            break
    
    originir = convert_qprog_to_originir(prog, ctx.machine)
    
    # normed_x = x / ||x|| => x = normed_x * ||x||
    if ratio == 0:
        return normed_x.tolist(), originir
    else:
        return (normed_x * ratio).tolist(), originir


# In[46]:


HHL(A, b)


# In[ ]:




