import tensorflow as tf
import numpy as np

batch_size = 64
embedding_dim = 10  # dimension of each word
num_hidden = 100  # number of hidden units in LSTM
att_num_hidden = 30
learning_rate = 0.05
momentum = 0.9
epoch = 50

en_seq_length = 15
de_seq_length = en_seq_length

en_vocab_size = 10  # Total vocab size including <eos> and <pad>
de_vocab_size = 10  # Total vocab size including <eos> and <pad>
num_layers = 3


# Data sequence preparation
eos = 1
pad = 2
unk = 3


def get_batch(batch_size, i):
    # TODO: Instead of picking random sequence lookup word ids/vectors from dictionary
    X = [np.random.choice(en_vocab_size-4, size=(np.random.randint(4, en_seq_length-1)))
         for _ in range(batch_size)]
    L = [len(x) for x in X]
    max_l = max(L) + 1
    # Add <eos> : id = vocab_size - 2
    X = [np.append(x, en_vocab_size-2) for x in X]
    # Add <pad> : id = vocab_size - 1
    X = [np.append(x, [en_vocab_size-1] * (max_l - len(x))) for x in X]
    # For testing purposes make output sequence equals to input sequence
    Y = [x[:de_seq_length] // 2 for x in X]
    D = [np.insert(x, 0, de_vocab_size - 1)[:-1] for x in Y]
    # Dimshuffle to seq_length * batch_size
    return X, Y, D, L


# Input and output sequences
pl_inputs = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='encoder_inputs')
pl_labels = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='decoder_targets')
pl_decoder = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='decoder_inputs')
pl_length = tf.placeholder(shape=(batch_size), dtype=tf.int32, name='sequence_lengths')

encoder_inputs = tf.one_hot(pl_inputs, en_vocab_size)
decoder_inputs = tf.one_hot(pl_decoder, de_vocab_size)

# Bidirectional layer uses bidirectional_dynamic_rnn
forward_cell = tf.contrib.rnn.LSTMCell(num_hidden/2)
backward_cell = tf.contrib.rnn.LSTMCell(num_hidden/2)
biout, bi_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, encoder_inputs, dtype=tf.float32, time_major=False)
outputs_concat = tf.concat([biout[0], biout[1]], 2)

# Encoder takes the bidirectional output. Uses dynamic_rnn
encoder_cells = []
for _ in range(num_layers):
    cell = tf.contrib.rnn.LSTMCell(num_hidden)
    encoder_cells.append(cell)

encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, outputs_concat, dtype=tf.float32, time_major=False, scope="plapl_encoder")

# Decoder also uses dynamic_rnn
decoder_cells = []
for _ in range(num_layers):
    cell = tf.contrib.rnn.LSTMCell(num_hidden)
    decoder_cells.append(cell)

# attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(att_num_hidden, encoder_outputs)
# decoder_cells[-1] = tf.contrib.seq2seq.AttentionWrapper(
#     decoder_cells[-1], attention_mechanism, name="attention")

decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)

training_helper = tf.contrib.seq2seq.TrainingHelper(
    inputs=decoder_inputs, sequence_length=pl_length, time_major=False, name='training_helper')

decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, training_helper, encoder_final_state)

# Dynamic decoding
max_decoder_length = tf.reduce_max(pl_length)
decoder_out, final_state, final_seq_len = tf.contrib.seq2seq.dynamic_decode(
    decoder, output_time_major=False)

output = tf.contrib.layers.fully_connected(decoder_out[0], de_vocab_size, activation_fn=tf.nn.sigmoid)
final_logit = tf.log(tf.clip_by_value(output, 1e-11, 1.0))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=pl_labels, logits=final_logit))
trapl_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=momentum).minimize(loss)

init = tf.global_variables_initializer()

# Add summary ops to collect
tf.summary.scalar("loss", loss)
merged_summary_op = tf.summary.merge_all()

print("Start training")

saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter("./logs", sess.graph)
    for iteration in range(epoch):
        avg_cost = 0.
        total_batch = 10  # It should be dependednt on the training data size
        for i in range(total_batch):
            x, y, de, ln = get_batch(batch_size, i)
            feed_dict = {pl_inputs: x, pl_labels: y, pl_decoder: de, pl_length: ln}
            sess.run(trapl_op, feed_dict=feed_dict)

            # Compute the average loss OPTIONAL
            avg_cost += sess.run(loss, feed_dict=feed_dict)/total_batch
            summary_str = sess.run(merged_summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, iteration*total_batch + 1)
        print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    saver.save(sess, 'savedmodels/model_num_seq2seq')

    print("Tuning Completed!")

    x, y, d, ln = get_batch(batch_size, i)
    feed_dict = {pl_inputs: x, pl_labels: y, pl_decoder: d, pl_length: ln}
    results = sess.run(output, feed_dict=feed_dict)
    Y_out = [logits_t.argmax(axis=1) for logits_t in results]

    print(x.T)
    print(np.array(Y_out).T)
