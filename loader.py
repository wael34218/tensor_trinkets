import tensorflow as tf
import numpy as np

batch_size = 64
embedding_dim = 10  # dimension of each word
num_hidden = 100  # number of hidden units in LSTM
learning_rate = 0.05
momentum = 0.9
epoch = 101
en_seq_length = 10
de_seq_length = 5
en_vocab_size = 8  # Total vocab size including <eos> and <pad>
de_vocab_size = 4  # Total vocab size including <eos> and <pad>
eos = 1
pad = 2
unk = 3

# Input and output sequences
# TODO: Bucketing
# TODO: Use word embedding vectors
enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="src%i" % t) for t in range(en_seq_length)]
labels = [tf.placeholder(tf.int32, shape=(None,), name="trg%i" % t) for t in range(de_seq_length)]
dec_inp = ([tf.zeros_like(labels[0], dtype=tf.int32, name="GO")] + labels[:-1])

# Neural Network Layers
with tf.name_scope("Seq2Seq") as scope:
    W1 = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]
    cell = tf.nn.rnn_cell.GRUCell(num_hidden)  # Can also use BasicLSTMCell
    outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(
        enc_inp, dec_inp, cell, en_vocab_size, de_vocab_size, embedding_dim)

with tf.name_scope("cross_entropy") as scope:
    loss = tf.nn.seq2seq.sequence_loss(outputs, labels, W1, de_vocab_size)
    magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))

with tf.name_scope("momentum_optimizer") as scope:
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)


def get_batch(batch_size, i):
    # TODO: Instead of picking random sequence lookup word ids/vectors from dictionary
    X = [np.random.choice(en_vocab_size-3, size=(np.random.randint(4, en_seq_length-1)))
         for _ in range(batch_size)]
    # Add <eos> : id = vocab_size - 2
    X = [np.append(x, en_vocab_size-2) for x in X]
    # Add <pad> : id = vocab_size - 1
    X = [np.append(x, [en_vocab_size-1] * (en_seq_length - len(x))) for x in X]
    # For testing purposes make output sequence equals to input sequence
    Y = [x[:de_seq_length] // 2 for x in X]

    # Dimshuffle to seq_length * batch_size
    X = np.array(X).T
    Y = np.array(Y).T
    return X, Y

saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    # new_saver = tf.train.import_meta_graph('savedmodels/model_num_seq2seq-9.meta')
    # new_saver.restore(sess, tf.train.latest_checkpoint('savedmodels/'))
    # graph = tf.get_default_graph()
    saver.restore(sess, 'savedmodels/model_num_seq2seq')
    print("Model restored.")

    batch_x, batch_y = get_batch(10, 1)
    test_feed = {enc_inp[t]: batch_x[t] for t in range(en_seq_length)}
    test_feed.update({labels[t]: batch_y[t] for t in range(de_seq_length)})
    dec_outputs_batch = sess.run(outputs, test_feed)
    Y_out = [logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]

    print(batch_x.T)
    print(np.array(Y_out).T)
