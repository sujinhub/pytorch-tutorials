import torch
import numpy as np


# [Tensor Initialization]

# 1. Creating "directly" from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# 2. Creating from NumPy Arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 3. Creating from another Tensor
x_ones = torch.ones_like(x_data) # preserves its properties
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overwrites its properties
print(f"Random Tensor: \n {x_rand} \n")

# 4. Creating using random or constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor}\n")
print(f"Ones Tensor: \n {ones_tensor}\n")
print(f"Zeros Tensor: \n {zeros_tensor}\n")


# [Tensor Attributes]
# These refer to its shape, datatype, and the device on which it is stored.

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on : {tensor.device}\n")


# [Tensor Operations]
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}\n")
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"{t1}\n")

y1 = tensor @ tensor.T # matrix multiplication
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y1, y2, y3, '\n', sep='\n') # y1=y2=y3

agg = tensor.sum()
agg_item = agg.item()
print(agg, agg_item, type(agg_item), '\n', sep='\n')

print(f"{tensor}\n")
tensor.add_(5)
print(tensor, '\n')

""" NOTE : "The in-place operation can save some memory,
but it immediately discards the history,
which can cause problems in calculating derivatives.
Therefore, it is not recommended to use in-place operations." """


# [NumPy Bridge]
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy() # Tensor to NumPy
print(f"n: {n}\n")

n = np.ones(5)
t = torch.from_numpy(n) # Numpy to Tensor

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
