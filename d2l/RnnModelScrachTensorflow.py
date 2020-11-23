from d2l import tensorflow as d2l
import math
import numpy as np
import tensorflow as tf

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


class RnnModelScrach:
    def __init__(self, num_hiddens, vocab_size):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = self._get_params()

    def _get_params(self):
        def init_weight(shape):
            w = tf.Variable(tf.random.normal(shape=shape, stddev=0.01), trainable=True)
            return w

        # Hidden layer parameters
        W_xh = init_weight((self.vocab_size, self.num_hiddens))
        W_hh = init_weight((self.num_hiddens, self.num_hiddens))
        b_h = tf.Variable(tf.zeros(self.num_hiddens), dtype=tf.float32, trainable=True)
        # Hidden layer parameters
        W_ho = init_weight((self.num_hiddens, self.vocab_size))
        b_o = tf.Variable(tf.zeros(self.vocab_size), dtype=tf.float32, trainable=True)
        params = [W_xh, W_hh, b_h, W_ho, b_o]
        return params

    def _init_state(self, batch_size):
        return tf.Variable(tf.zeros((batch_size, self.num_hiddens), dtype=tf.float32), trainable=True)

    def __call__(self, x, state=None):
        bs = x.shape[0]
        x = tf.transpose(x)

        x = tf.one_hot(x, self.vocab_size)
        if state is None:
            state = self._init_state(bs)
        x = tf.cast(x, dtype=tf.float32)
        Y, state = self.forward(x, state)
        return Y, state

    def forward(self, x, state):
        ys = []
        W_xh, W_hh, b_h, W_ho, b_o = self.params
        for xi in x:
            state = tf.matmul(xi, W_xh) + tf.matmul(state, W_hh) + b_h
            state = tf.tanh(state)
            y = tf.matmul(state, W_ho) + b_o
            ys.append(y)
        Y = tf.concat(ys, axis=0)
        return Y, state

    def predict(self, vocab=vocab, prefix='generated ', num_tokens=50):
        results = [vocab[prefix[0]]]
        state = self._init_state(batch_size=1)

        def get_input_token_id():
            token_id = tf.constant(results[-1])
            token_id = tf.reshape(token_id, (1, 1)).numpy()  # todo
            # print(token_id)
            return token_id

        for y in prefix[1:]:
            _, state = self(get_input_token_id(), state)
            results.append(vocab[y])
        for _ in range(num_tokens):
            y, state = self(get_input_token_id(), state)
            results.append(int(y.numpy().argmax(axis=1).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in results])


def grad_clipping(grads, theta):  # @save
    """Clip the gradient."""
    theta = tf.constant(theta, dtype=tf.float32)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                            for grad in grads))
    norm = tf.cast(norm, tf.float32)
    new_grad = []
    if tf.greater(norm, theta):
        for grad in grads:
            new_grad.append(grad * theta / norm)
    else:
        for grad in grads:
            new_grad.append(grad)
    return new_grad


def train_epoch_ch8(model, train_iter, loss, updater, params, use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = model._init_state(batch_size=X.shape[0])
        with tf.GradientTape(persistent=True) as g:
            g.watch(params)
            y_hat, state = model(X, state)
            y = tf.reshape(Y, (-1))
            l = loss(y, y_hat)
        grads = g.gradient(l, params)
        grads = grad_clipping(grads, 0.5)
        updater.apply_gradients(zip(grads, params))

        # Keras loss by default returns the average loss in a batch
        # l_sum = l * float(tf.size(y).numpy()) if isinstance(
        #     loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l * tf.size(y).numpy(), tf.size(y).numpy())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# @save
def train_ch8(model, train_iter, vocab, lr, num_epochs, strategy, use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    params = model.params  # ---------------------------------------------------------------------------------------------------------
    with strategy.scope():

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        updater = tf.keras.optimizers.Adam(lr, amsgrad=True)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # def predict(self,vocab=vocab,prefix='generated ',num_tokens=50):
    predict = lambda prefix: model.predict(vocab=vocab, prefix=prefix, num_tokens=50)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            model, train_iter, loss, updater, params, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    device = d2l.try_gpu()._device_name
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

    device_name = d2l.try_gpu()._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    num_epochs, num_hiddens, lr = 500, 512, 0.0001
    model=RnnModelScrach(num_hiddens,len(vocab))
    train_ch8(model, train_iter, vocab,lr, num_epochs, strategy)