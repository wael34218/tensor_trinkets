import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)

# hyperparameters
lr = 0.001
epoch = 1
batch_size = 100

n_inputs = 28   # MNIST data input (img shape: 28*28)
n_steps = 28    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)
n_layers = 1

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], "Pixels")
y = tf.placeholder(tf.float32, [None, n_classes], "Classes")

with tf.name_scope("Hidden1") as scope:
    W1 = tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
    B1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]))
    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(x, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    Y1 = tf.matmul(X, W1) + B1
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    Y1 = tf.reshape(Y1, [-1, n_steps, n_hidden_units])

with tf.name_scope("LSTM") as scope:
    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.

    cell = tf.nn.rnn_cell.GRUCell(n_hidden_units)  # Can also use BasicLSTMCell
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layers)
    Y2, final_state = tf.nn.dynamic_rnn(cell, Y1, dtype=tf.float32, time_major=False)

with tf.name_scope("Output") as scope:
    B2 = tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    W2 = tf.Variable(tf.random_normal([n_hidden_units, n_classes]))

    # unpack to list [(batch, outputs)..] * steps
    Y2 = tf.unpack(tf.transpose(Y2, [1, 0, 2]))    # states is the last outputs
    # Y2[-1] to take the final output of the sequence
    y_pred = tf.matmul(Y2[-1], W2) + B2

with tf.name_scope("Cost") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))

with tf.name_scope("Optimizer") as scope:
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

init = tf.global_variables_initializer()

w1 = tf.summary.histogram("WHidden", W1)
b1 = tf.summary.histogram("BHidden", B1)
w2 = tf.summary.histogram("WOut", W2)
b2 = tf.summary.histogram("BOut", B2)
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter("./logs", sess.graph)
    for iteration in range(epoch):
        total_batch = int(mnist.train.num_examples/batch_size)
        for _ in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + 1)
        print(sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}))

    print("Tuning Completed!")

    predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))

    print("Last Batch Training Accuracy: ", accuracy.eval({x: batch_xs, y: batch_ys}))

    total_batch = int(mnist.test.num_examples/batch_size)
    batch_xs = mnist.test.images.reshape([-1, n_steps, n_inputs])
    print("Accuracy: ", accuracy.eval({x: batch_xs, y: mnist.test.labels}))
