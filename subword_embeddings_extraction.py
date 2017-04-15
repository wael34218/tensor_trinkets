'''
A way to extract n-gram vector representation from word2vec embeddings
'''
import numpy as np
import tensorflow as tf
from functools import reduce

subword_count = 15
hidden_units = 20
embedding_size = 16
learning_rate = 0.01
momentum = 0.1
max_subwords = 15
min_subwords = 12
batch_size = 256
epoch = 35
total_batches = 50

# Input
sub_words = tf.placeholder(shape=(None, subword_count), dtype=tf.float32)

# Output
semantic = tf.placeholder(shape=(None, embedding_size), dtype=tf.float32)

# Network
with tf.name_scope("Full") as scope:
    W1 = tf.Variable(initial_value=tf.random_uniform([subword_count, embedding_size], -0.1, 0.1),
                     name="W1", dtype=tf.float32)
    Y1 = tf.matmul(sub_words, W1)

with tf.name_scope("RMSE") as scope:
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y1, semantic))))
    tf.summary.scalar("Loss", loss)

with tf.name_scope("optimizer") as scope:
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

# Create or load word vectors:
vecs = []  # This should be the word vector representation
one_hot = []  # This is one hot representaiton of all the n-gram subwords
for sv in range(subword_count):
    init_vec = [0.] * embedding_size
    init_vec[sv] = 1.0
    vecs.append(init_vec)

    init_vec = [0] * subword_count
    init_vec[sv] = 1
    one_hot.append(init_vec)


def next_feed():
    words = [np.random.choice(subword_count, size=(np.random.randint(min_subwords, max_subwords)))
             for _ in range(batch_size)]

    x = np.array([reduce(lambda a, b: a+b, map(lambda x: np.array(one_hot[x]), w)) for w in words])
    y = np.array([reduce(lambda a, b: a+b, map(lambda x: np.array(vecs[x]), w)) for w in words])
    return {sub_words: x, semantic: y}, words

init = tf.global_variables_initializer()
merged_summary_op = tf.summary.merge_all()

# Launch graph
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter("./logs", sess.graph)
    for iteration in range(epoch):
        avg_cost = 0.
        for i in range(total_batches):
            feed_dict, words = next_feed()
            sess.run(optimizer, feed_dict=feed_dict)
            avg_cost += sess.run(loss, feed_dict=feed_dict)/total_batches
            summary_str = sess.run(merged_summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, iteration*total_batches + 1)
        print("\nIteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Subword IDs:" + str(words[0]))
        print("Target Vector: " + str(feed_dict[semantic][0]))
        predict = sess.run(Y1, feed_dict)
        print(["%.2f" % x for x in predict[0]])
