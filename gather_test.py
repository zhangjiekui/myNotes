import torch
test1 = {"input":torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]]),
         "index": torch.LongTensor([0,2]).view(-1,1),
         'dim':1
         }

test4 = {"input":torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]]),
         "index": torch.LongTensor([[0,1]]),
         'dim':0,
        }

input=torch.tensor([[1,2,3],[4,5,6]])
index1=torch.tensor([[0,1,1],[0,0,1]])

test2 = {"input":input,
         "index":index1,
         'dim':0
         }

test3 = {"input":input,
         "index":index1,
         'dim':1
         }


tests = [test1,test2,test3,test4]

def torch_gather_test(input,index,dim):
    output_shape=index.shape
    output=torch.ones(output_shape)*-10
    # print(output.shape)
    if dim == 0:
        print("--------------dim0")
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                # print(f"my_{output=}")
                _i = index[i,j]
                output[i,j] = input[_i,j]
                # print(f"{i=},{_i=},{j=},{output[i,j].tolist()=}")
        print(f"my_{output.tolist()= }")
        print("gather           = ",torch.gather(input,dim,index).to(dtype=output.dtype).tolist())


    elif dim == 1:
        print("--------------dim1")
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                # print(f"my_{output=}")
                _j = index[i,j]
                output[i,j] = input[i,_j]
                # print(f"{i=},{_j=},{j=},{output[i,j].tolist()=}")
        print(f"my_{output.tolist()= }")
        # print(f"orignal_{result.tolist()=}")
        print("gather           = ",torch.gather(input,dim,index).to(dtype=output.dtype).tolist())
    else:
        pass

for test in tests:
    input = test['input']
    index = test['index']
    dim = test['dim']
    torch_gather_test(input,index,dim)