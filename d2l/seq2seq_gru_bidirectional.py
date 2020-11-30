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

from mxnet.gluon import nn,rnn
from mxnet import init,gluon,autograd

class Encoder(nn.Block):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def forward(self,X,*args):
        raise NotImplementedError
class Decoder(nn.Block):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def init_state(self,enc_outputs,*args):
        raise NotImplementedError
    def forward(self,X,state):
        raise NotImplementedError
class EncoderDecoder(nn.Block):
    def __init__(self,encoder,decoder,**kwargs):
        super().__init__(**kwargs)
        self.encoder=encoder
        self.decoder=decoder
    def forward(self,enc_x,dec_x,*args):
        enc_outputs=self.encoder(enc_x,*args)
        dec_state=self.decoder.init_state(enc_outputs,*args)
        return self.decoder(dec_x,dec_state)
class Seq2SeqEncoder(Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,bidirectional=True,dropout=0,**kwargs):
        super().__init__(**kwargs)
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=rnn.GRU(hidden_size=num_hiddens,num_layers=num_layers,bidirectional=bidirectional,dropout=dropout)
    def forward(self,X): #(bs,num_steps)
        X=self.embedding(X) #(bs,num_steps,embed_size)
        X=X.swapaxes(0,1)   #(num_steps,bs,embed_size)
        # In RNN models, the first axis corresponds to time steps
        state=self.rnn.begin_state(batch_size=X.shape[1],ctx=X.ctx)
        # `output` shape:   (`num_steps` , `batch_size`, `num_hiddens`),  If `bidirectional` is True, output shape will instead be `(sequence_length, batch_size, 2*num_hidden)`
        # `state[0]` shape: (`num_layers`, `batch_size`, `num_hiddens`),  If `bidirectional` is True, shape will instead be `(2*num_layers, batch_size, num_hidden)`.
        output,state=self.rnn(X,state)
        if self.rnn._dir==2: # 说明是双向网络
            #需要对state的前向、后向进行连接处理
            s=state[0] #(2*num_layers, batch_size, num_hidden)
            fwd=s[0:s.shape[0]:2]  #s[start:end:step]   #从start开始,以step为步长直到end结束。
            bwd=s[1:s.shape[0]:2]
            s=mxnp.concatenate([fwd,bwd],axis=-1) # (num_layers, batch_size, 2*num_hidden)
            state=(s,)
        # output (sequence_length, batch_size, _bir*num_hidden)
        # state:tuple (s,) 其中s (num_layers,      batch_size, _bir*num_hidden)
        return output,state

class Seq2SeqDecoder(Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,dropout=0,bidirectional_encoder=True, **kwargs):
        super().__init__(**kwargs)
        if bidirectional_encoder:
            num_hiddens*=2
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1] #返回的是hidden_state (s,)

    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).swapaxes(0, 1)
        # `context` shape: (`batch_size`, `num_hiddens`)
        context = state[0][-1]
        #print(f"context.shape:{context.shape}")
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = mxnp.broadcast_to(context, (X.shape[0], context.shape[0], context.shape[1]))
        # X_and_context shape: `num_steps`, `batch_size`, (embed_size + bi*num_hiddens)
        X_and_context = mxnp.concatenate((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).swapaxes(0, 1)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state[0]` shape: (`num_layers`, `batch_size`, `bidirectional*num_hiddens`)
        return output, state

def build_encoder_decoder(vocab_size,embed_size,num_hiddens,num_layers,bidirectional_encoder=True,dropout=0,**kwargs):
    encoder=Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers,bidirectional=bidirectional_encoder,dropout=0,**kwargs)
    decoder=Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers,bidirectional_encoder=bidirectional_encoder,dropout=0, **kwargs)
    en_decoder=EncoderDecoder(encoder,decoder)
    en_decoder.initialize()
    return en_decoder

X = mxnp.zeros((4, 7))
print(f'X.shape:{X.shape}')
en_decoder=build_encoder_decoder(vocab_size=101, embed_size=80, num_hiddens=32,num_layers=8,bidirectional_encoder=False)
output, state = en_decoder(X,X)
print(f'bidirectional_encoder=False. output.shape:{output.shape},state[0].shape:{state[0].shape}')
en_decoder=build_encoder_decoder(vocab_size=101, embed_size=80, num_hiddens=32,num_layers=8,bidirectional_encoder=True)
output, state = en_decoder(X,X)
print(f'bidirectional_encoder= True. output.shape:{output.shape},state[0].shape:{state[0].shape}')