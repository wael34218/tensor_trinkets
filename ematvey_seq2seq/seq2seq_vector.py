import numpy as np
import tensorflow as tf
import helpers

tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1

vocab_size = 10
embedding_size = 2

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None, embedding_size),
                                 dtype=tf.float32, name='decoder_targets')

decoder_inputs = tf.placeholder(shape=(None, None, embedding_size),
                                dtype=tf.float32, name='decoder_inputs')

embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                         dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    dtype=tf.float32, time_major=True,
)
del encoder_outputs

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs,
    initial_state=encoder_final_state,
    dtype=tf.float32, time_major=True, scope="plain_decoder",
)

output_embeddings = tf.contrib.layers.linear(decoder_outputs, embedding_size)
loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(decoder_targets, output_embeddings))))

train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

batch_size = 100

batches = helpers.random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)


def next_feed():
    batch = next(batches)
    encoder_inputs_, _ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] for sequence in batch]
    )
    decoder_inputs_, _ = helpers.batch(
        [[EOS] + (sequence) for sequence in batch]
    )
    dt = np.array([[[y/2]*embedding_size for y in x] for x in decoder_targets_])
    di = np.array([[[y/2]*embedding_size for y in x] for x in decoder_inputs_])
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: di,
        decoder_targets: dt,
    }

max_batches = 20001
batches_in_epoch = 1000

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)

        if batch == 0 or batch % batches_in_epoch == 0:
            predict_ = sess.run(output_embeddings, fd)
            print("Loss :" + str(l))
            print("Inputs: " + str(fd[encoder_inputs].T[0]))
            print("Predicted: " + str(predict_[:, 0, :]))

except KeyboardInterrupt:
    print('training interrupted')
