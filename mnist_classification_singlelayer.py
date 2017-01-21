import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/mnist", one_hot=True, validation_size=0)

learning_rate = 0.01
training_iteration = 30
batch_size = 100

# Input ... never initialized and contains no data
x = tf.placeholder("float", [None, 784], name="Pixles")
y = tf.placeholder("float", [None, 10], name="TrueLabels")

# If images were in 28*28 format then we have to do reshape:
# x = tf.reshape(X, [-1, 784]) => -1 means there is only one solution figure it out
# -1 will end up to be the number of images in the dataset

# Model weights
W = tf.Variable(initial_value=tf.zeros([784, 10]), name="Weights")
b = tf.Variable(initial_value=tf.zeros([10]), name="biases")
# training means computing those variables

with tf.name_scope("model_soft_Wx_b") as scope:
    # Logistic Regression()
    y_pred = tf.nn.softmax(tf.matmul(x, W) + b)

# Add summary ops to collect
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)


with tf.name_scope("cost_function_cross_entropy") as scope:
    # Cross entropy
    # cost_function = -tf.reduce_sum(y*tf.log(y_pred))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y)
    cost_function = tf.reduce_sum(cross_entropy)
    # Create a summary to monitor the cost function
    tf.summary.scalar("cost_function", cost_function)

# Optimizer
with tf.name_scope("optimizer_GD") as scope:
    # Gradient descent: minimize cost function
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()
merged_summary_op = tf.summary.merge_all()

# Launch graph
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter("./logs", sess.graph)
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute the average loss OPTIONAL
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + 1)
        print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Tuning Completed!")

    predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
