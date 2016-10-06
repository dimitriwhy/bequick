#!/usr/bin/env python
import math
import tensorflow as tf
tf.set_random_seed(1234)


class Model(object):
    def __init__(self,
                 algorithm,
                 form_size,
                 form_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 max_steps,
                 batch_size,
                 regularizer):
        self.algorithm = algorithm
        self.form_size = form_size
        self.form_dim = form_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layers
        self.max_steps = max_steps
        self.batch_size = batch_size

        # PLACEHOLDER: input and output
        self.X = tf.placeholder(tf.int32, (batch_size, max_steps), name="X")
        # shape(X) => (50, 32)
        self.Y = tf.placeholder(tf.int32, (batch_size, max_steps), name="Y") # use the sparse version
        # shape(Y) => (50, 32)
        self.early_stop = tf.placeholder(tf.int32, (batch_size,), name="len")

        # EMBEDDING in CPU
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            width = math.sqrt(6. / (form_size + form_dim))
            self.form_emb = tf.Variable(tf.random_uniform([self.form_size, self.form_dim], -width, width),
                                        name="form_emb")
            inputs = tf.nn.embedding_lookup(self.form_emb, self.X)
            # shape(inputs) => (batch_size, max_stop, form_dim)
            inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, max_steps, inputs)]
            # shape(inputs[0]) => (batch_size, max_stop)

        # RNN
        fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
        bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
        fw_stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell] * n_layers, state_is_tuple=True)
        bw_stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell] * n_layers, state_is_tuple=True)

        outputs, _, _ = tf.nn.bidirectional_rnn(fw_stacked_lstm_cell, bw_stacked_lstm_cell,
                                                inputs,
                                                sequence_length=self.early_stop,
                                                dtype=tf.float32)

        output = tf.reshape(tf.concat(1, outputs), [-1, self.hidden_dim * 2])
        width = math.sqrt(6. / (self.hidden_dim * 2 + self.output_dim))
        self.W0 = tf.Variable(tf.random_uniform([self.hidden_dim * 2, self.output_dim], -width, width), name="W0")
        self.b0 = tf.Variable(tf.zeros([self.output_dim]), name="b0")
        output = tf.nn.softmax(tf.matmul(output, self.W0) + self.b0)
        self.pred = tf.reshape(output, (self.batch_size, -1, self.output_dim))

        # LOSS
        reg = regularizer * (tf.nn.l2_loss(self.W0) + tf.nn.l2_loss(self.b0))
        self.loss = (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.pred, self.Y)) + reg)

        if algorithm == "adam":
            self.optm = tf.train.AdamOptimizer().minimize(self.loss)
        elif algorithm == "adagrad":
            self.optm = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(self.loss)
        elif algorithm == "adadelta":
            self.optm = tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(self.loss)
        else:
            # GRADIENTS AND CLIPPING
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(0.01, global_step, 100000, 0.96)
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5.)
            self.optm = opt.apply_gradients(zip(grads, tvars), global_step=global_step)

    def init(self):
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def initialize_word_embeddings(self, indices, matrix):
        _indices = [tf.to_int32(i) for i in indices]
        self.session.run(tf.scatter_update(self.form_emb, _indices, matrix))

    def train(self, X, Y, steps):
        _, cost = self.session.run(
                [self.optm, self.loss],
                feed_dict = {self.early_stop: steps, self.X: X, self.Y: Y}
                )
        return cost

    def classify(self, X, steps):
        return self.session.run(self.pred,
                feed_dict = {self.early_stop: steps, self.X: X}
                )
