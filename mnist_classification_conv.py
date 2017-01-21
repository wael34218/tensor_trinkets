import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/mnist", one_hot=True, validation_size=0)

lr = tf.placeholder(tf.float32, name="LearningRate")
epoch = 4
batch_size = 128
learning_rate = 0.003
max_lr = 0.003
min_lr = 0.0001
decay_speed = 2000.0

# Input ... never initialized and contains no data
x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="Pixles")
y = tf.placeholder(tf.float32, [None, 10], name="TrueLabels")

# If images were in 28*28 format then we have to do reshape:
# x = tf.reshape(X, [-1, 784]) => -1 means there is only one solution figure it out
# -1 will end up to be the number of images in the dataset

K = 4
L = 8
M = 12
N = 200

# Neural Network Layers
with tf.name_scope("Conv1") as scope:
    W1 = tf.Variable(initial_value=tf.truncated_normal([5, 5, 1, K], stddev=0.1))
    B1 = tf.Variable(initial_value=tf.ones([K])/10)
    Y1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)

with tf.name_scope("Conv2") as scope:
    W2 = tf.Variable(initial_value=tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(initial_value=tf.ones([L])/10)
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)

with tf.name_scope("MaxPool") as scope:
    Y2B = tf.nn.max_pool(Y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("Conv3") as scope:
    W3 = tf.Variable(initial_value=tf.truncated_normal([5, 5, L, M], stddev=0.1))
    B3 = tf.Variable(initial_value=tf.ones([M])/10)
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2B, W3, strides=[1, 2, 2, 1], padding='SAME') + B3)
    YY = tf.reshape(Y3, shape=[-1, 7*7*M])

with tf.name_scope("FullyConnected") as scope:
    W4 = tf.Variable(initial_value=tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
    B4 = tf.Variable(initial_value=tf.ones([N])/10)
    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)

with tf.name_scope("Softmax") as scope:
    W5 = tf.Variable(initial_value=tf.truncated_normal([N, 10], stddev=0.1))
    B5 = tf.Variable(initial_value=tf.zeros([10])/10)
    y_pred = tf.nn.softmax(tf.matmul(Y4, W5) + B5)

with tf.name_scope("cross_entropy") as scope:
    # Cross entropy
    # Option 1:
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y)
    # cost_function = tf.reduce_sum(cross_entropy)

    # Option 2:
    cost_function = -tf.reduce_sum(y*tf.log(y_pred))

    # Create a summary to monitor the cost function
    tf.summary.scalar("cost_function", cost_function)


# Add summary ops to collect
w1 = tf.summary.histogram("WConv1", W1)
b1 = tf.summary.histogram("BConv1", B1)
w2 = tf.summary.histogram("WConv2", W2)
b2 = tf.summary.histogram("BConv2", B2)
w3 = tf.summary.histogram("WConv3", W3)
b3 = tf.summary.histogram("BConv3", B3)
w4 = tf.summary.histogram("WFull4", W4)
b4 = tf.summary.histogram("BFull4", B4)
w5 = tf.summary.histogram("WOut5", W5)
b5 = tf.summary.histogram("BOut5", B5)

# Optimizer
with tf.name_scope("optimizer_Adam") as scope:
    # Option 1: Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    # Option 2: Adam Optimizer
    # optimizer = tf.train.AdamOptimizer(lr).minimize(cost_function)

init = tf.global_variables_initializer()
merged_summary_op = tf.summary.merge_all()

# Launch graph
with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter("./logs", sess.graph)
    for iteration in range(epoch):
        learning_rate = min_lr + (max_lr - min_lr) * math.exp(-iteration/decay_speed)
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([-1, 28, 28, 1])
            # Fit training using batch data
            # For Gradient Descent Optimizer
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # For Adam Optimizer
            # sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, lr: learning_rate})

            # Compute the average loss OPTIONAL
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + 1)
        print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Tuning Completed!")

    predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy: ", accuracy.eval({x: mnist.test.images.reshape([-1, 28, 28, 1]),
                                       y: mnist.test.labels}))
