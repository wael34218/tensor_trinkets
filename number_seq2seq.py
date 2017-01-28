import tensorflow as tf
import numpy as np

seq_length = 10
batch_size = 64
vocab_size = 7  # Total vocab size including <eos> and <pad>
embedding_dim = 10  # dimension of each word
memory_dim = 100
learning_rate = 0.05
momentum = 0.9
epoch = 50

# Input and output sequences
# TODO: Bucketing
# TODO: Use word embedding vectors
enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="src%i" % t) for t in range(seq_length)]
labels = [tf.placeholder(tf.int32, shape=(None,), name="trg%i" % t) for t in range(seq_length)]
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=tf.int32, name="GO")] + enc_inp[:-1])

# Neural Network Layers
with tf.name_scope("Seq2Seq") as scope:
    W1 = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]
    cell = tf.nn.rnn_cell.GRUCell(memory_dim)  # Can also use BasicLSTMCell
    outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(
        enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_dim)

with tf.name_scope("cross_entropy") as scope:
    loss = tf.nn.seq2seq.sequence_loss(outputs, labels, W1, vocab_size)
    magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))

with tf.name_scope("momentum_optimizer") as scope:
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

init = tf.global_variables_initializer()

# Add summary ops to collect
tf.summary.scalar("loss", loss)
tf.summary.scalar("Magnitude at t=1", magnitude)
merged_summary_op = tf.summary.merge_all()
summary_op = tf.summary.merge_all()


def get_batch(batch_size):
    # TODO: Instead of picking random sequence lookup word ids/vectors from dictionary
    X = [np.random.choice(vocab_size-3, size=(np.random.randint(4, seq_length-1)))
         for _ in range(batch_size)]
    # Add <eos>
    X = [np.append(x, vocab_size-2) for x in X]
    # Add <pad>
    X = [np.append(x, [vocab_size-1] * (seq_length - len(x))) for x in X]
    # For testing purposes make output sequence equals to input sequence
    Y = X[:]

    # Dimshuffle to seq_length * batch_size
    X = np.array(X).T
    Y = np.array(Y).T
    return X, Y


with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter("./logs", sess.graph)
    for iteration in range(epoch):
        avg_cost = 0.
        total_batch = 10  # It should be dependednt on the training data size
        for i in range(total_batch):
            batch_x, batch_y = get_batch(batch_size)
            feed_dict = {enc_inp[t]: batch_x[t] for t in range(seq_length)}
            feed_dict.update({labels[t]: batch_y[t] for t in range(seq_length)})
            sess.run(optimizer, feed_dict=feed_dict)

            # Compute the average loss OPTIONAL
            avg_cost += sess.run(loss, feed_dict=feed_dict)/total_batch
            summary_str = sess.run(merged_summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, iteration*total_batch + 1)
        print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Tuning Completed!")

    batch_x, batch_y = get_batch(10)
    test_feed = {enc_inp[t]: batch_x[t] for t in range(seq_length)}
    test_feed.update({labels[t]: batch_y[t] for t in range(seq_length)})
    dec_outputs_batch = sess.run(outputs, test_feed)
    Y_out = [logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]
    print(batch_x.T)
    print(np.array(Y_out).T)
