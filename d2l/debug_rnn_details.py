
import torch as tc
import tensorflow as tf
import mxnet as mx
from mxnet import np as mxnp
from mxnet import npx as npx
npx.set_np()
# import numpy as np

from d2l import mxnet as mxd2l  # Use MXNet as the backend
from d2l import torch as tcd2l  # Use PyTorch as the backend
from d2l import tensorflow as tfd2l  # Use TensorFlow as the backend

print(tc.__version__,tf.__version__,mx.__version__)

import math
from mxnet import autograd,gluon
d2l=mxd2l


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal_init(shape):
        return mxnp.random.normal(scale=0.01, size=shape, ctx=device)

    W_xh = normal_init((num_inputs, num_hiddens))
    W_hh = normal_init((num_hiddens, num_hiddens))
    b_h = mxnp.zeros(num_hiddens, ctx=device)

    W_ho = normal_init((num_hiddens, num_outputs))
    b_o = mxnp.zeros(num_outputs, ctx=device)

    params = [W_xh, W_hh, b_h, W_ho, b_o]
    for param in params:
        param.attach_grad()
    return params

def init_rnn_state(batch_size,num_hiddens,device):
    '''
    初始化H0  ，hidden_state
    '''
    H0=mxnp.zeros((batch_size,num_hiddens),ctx=device)
    return (H0,)


def rnn(inputs, hidden_states, params):
    '''
    inputs shape: (num_steps[seq-len],batch_size,vocab_size)
    return:
          outputs,(H,)
    '''
    W_xh, W_hh, b_h, W_ho, b_o = params
    H, = hidden_states
    outputs = []
    hidden_states = []
    # X shape: (batch_size,vocab_size)
    print(f"rnn loops  {inputs.shape[0]} times along seq_length axis---------\n")
    i=1
    for X in inputs:  # 沿着num_steps(sequence length)循环
        print(f"loops  {i} times \n")
        i+=1
        H = mxnp.dot(X, W_xh) + mxnp.dot(H, W_hh) + b_h
        H = mxnp.tanh(H)
        hidden_states.append(H)

        print(f"---rnn input(X,H) and weights' shape---------\n"
              f"   ---X.shape={X.shape},W_xh.shape={W_xh.shape}\n"
              f"   ---H.shape={H.shape},W_hh.shape={W_hh.shape},b_h.shape={b_h.shape}\n"
              f"   ---W_ho.shape={W_ho.shape},b_o.shape={b_o.shape}\n")

        Y = mxnp.dot(H, W_ho) + b_o
        outputs.append(Y)
        print(f"---rnn output's shape---------\n"
              f"   ---Y.shape={Y.shape},H.shape={H.shape}\n")
    Ys = mxnp.concatenate(outputs, axis=0)
    print(f"Final Ys.shape={Ys.shape}")
    return Ys, (H,), hidden_states, outputs


class RnnModelScrach:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)

        self.init_state = init_state
        self.forward_fn = forward_fn
        self.device = device

    def init_state_h0(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens, self.device)

    def __call__(self, X, state=None):
        X = npx.one_hot(X.T, self.vocab_size)  # return shape: (num_steps, batch_size,vocab_size)
        if state is None:
            batch_size = X.shape[1]
            state = self.init_state_h0(batch_size)
        return self.forward_fn(X, state, self.params)


def predict(prefix, num_preds, model, vocab):
    state = model.init_state_h0(batch_size=1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(mxnp.array(outputs[-1], ctx=model.device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _,state,_,_=model(get_input(),state)
        #_, state, _, _ = model(get_input()) ##state会更新，所以不能传入空
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state, _, _ = model(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1)))

    return ''.join([vocab.idx_to_token[i] for i in outputs])

num_hiddens = 52
vocab_size=28
model=RnnModelScrach(vocab_size,num_hiddens,d2l.try_gpu(),get_params=get_params,init_state=init_rnn_state,forward_fn=rnn)
X=d2l.reshape(mxnp.arange(10),(2,5))
Y, new_state,hidden_states_list,outputs_list = model(X.as_in_context(d2l.try_gpu()), None)
print("---end---\n"*3)

train_iter, vocab = d2l.load_data_time_machine(2, 6)
result=predict('time traveller ',10,model,vocab)

print(result)
#cqhwymvcqh

