import torch
a = torch.tensor([[1,5,62,54], 
                  [2,6,2,6], 
                  [2,65,2,6]])
print(a.shape) # torch.Size([3, 4])
out_0_column = torch.max(a, 0) #0是每列的最大值
print(out_0_column)
# torch.return_types.max(
# values=tensor([ 2, 65, 62, 54]),
# indices=tensor([1, 2, 0, 0]))
print("==="*10)
out_1_row = torch.max(a, 1) #1是每行的最大值
print(out_1_row)
# torch.return_types.max(
# values=tensor([62,  6, 65]),
# indices=tensor([2, 3, 1]))

print(a[:,0])

zero_tensor=torch.tensor([]).unsqueeze(1)

print(zero_tensor)


r=a[:,0]*zero_tensor
print(r)

non_zero_idx = torch.nonzero(r)
print(non_zero_idx)
print(non_zero_idx.numpy())
