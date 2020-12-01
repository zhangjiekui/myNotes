from mxnet import gluon
from mxnet import np
from d2l import mxnet as mxd2l
from seq2seq_gru_bidirectional import *
def train_s2s(model:gluon.nn.Block,data_iter,lr,num_epochs,tgt_vocab,device):
    trainer=gluon.Trainer(model.collect_params(),'adam',{'learning_rate':lr})
    loss=MaskedSoftmaxCELoss()
    animator=d2l.Animator(xlabel='epocs',ylabel='loss',xlim=[10,num_epochs])
    for epoch in range(num_epochs):
        timer=d2l.Timer()
        metric=d2l.Accumulator(2)
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.as_in_ctx(device) for x in batch]
            bos_id=tgt_vocab['<bos>']
            bos=np.array([bos_id]*Y.shape[0],ctx=device)
            bos=bos.reshape(-1,1)
            dec_input=np.concatenate([bos,Y[:,:-1]],axis=1)  # Teacher forcing  # bos+Y(除了最后一个字符)
            with autograd.record():
                Y_hat,_=model(X,dec_input)
                l=loss(Y_hat,Y,Y_valid_len)
            l.backward()
            d2l.grad_clipping(model,1)
            num_tokens=Y_valid_len.sum()
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
            if (epoch + 1) % 10 == 0:
                animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
def predict_s2s_ch9(model, src_sentence, src_vocab, tgt_vocab, num_steps,device):
    """Predict sequences (defined in Chapter 9)."""
    #src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    src_tokens = src_vocab[get_word_list(src_sentence.lower())] + [src_vocab['<eos>']]
    enc_valid_len = np.array([len(src_tokens)], ctx=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = np.expand_dims(np.array(src_tokens, ctx=device), axis=0)
    enc_outputs = model.encoder(enc_X, enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = np.expand_dims(np.array([tgt_vocab['<bos>']], ctx=device), axis=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(axis=2)
        pred = dec_X.squeeze(axis=0).astype('int32').item()
        # Once the end-of-sequence token is predicted, the generation of
        # the output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq))

if __name__=='__main__':
    batch_size, num_steps = 64, 16
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab, source, target = load_data_nmt(batch_size, num_steps)
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    model = build_encoder_decoder(src_vocab_size, tgt_vocab_size, embed_size, num_hiddens, num_layers,bidirectional_encoder=True, dropout=dropout)
    num_epochs_plus = 1
    train_s2s(model, train_iter, lr, num_epochs_plus, tgt_vocab, device)
    src_sentence = "找到汤姆。"
    p=predict_s2s_ch9(model, src_sentence, src_vocab, tgt_vocab, num_steps, device)
    print(p)
