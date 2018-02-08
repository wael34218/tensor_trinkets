import tensorflow as tf
import numpy as np
# from tensorflow.python import debug as tfdbg

batch_size = 32
embedding_dim = 10  # dimension of each word
num_hidden = 100  # number of hidden units in LSTM
att_num_hidden = 100
learning_rate = 0.001
momentum = 0.9
epoch = 200
attention = True

en_vocab_size = 10  # Total vocab size including <eos> and <pad>
de_vocab_size = 9  # Total vocab size including <eos> and <pad>
num_layers = 2

# Data sequence preparation
go = 0
eos = 1
pad = 2
unk = 3

en_seq_length = 7
de_seq_length = 7

beam_size = 3

# TODOS:
# 1- batch_size should not be defined in the network
# 2- Load and store model
# 3- Beamsearch fix
# 4- Attention fix pad problem
# 5- Residual connections
# 6- Dropout


def get_batch(batch_size, i):
    # TODO: Instead of picking random sequence lookup word ids/vectors from dictionary
    X = [4 + np.random.choice(en_vocab_size, size=(np.random.randint(4, en_seq_length)))
         for _ in range(batch_size)]
    Y = [2 + (x // 2) for x in X]

    XL = [len(x) + 1 for x in X]
    YL = [len(x) + 1 for x in Y]

    max_xl = max(XL)
    max_yl = max(YL)

    X = [np.append(x, eos) for x in X]
    X = [np.append(x, [pad] * (max_xl - len(x))) for x in X]

    Y = [np.append(x, eos) for x in Y]
    Y = [np.append(x, [pad] * (max_yl - len(x))) for x in Y]

    return X, Y, XL, YL


# Input and output sequences
pl_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='enc_inp')
pl_labels = tf.placeholder(shape=(None, None), dtype=tf.int32, name='dec_tar')
pla_inp_len = tf.placeholder(shape=(None, ), dtype=tf.int32, name='enc_seq_len')
pla_lab_len = tf.placeholder(shape=(None, ), dtype=tf.int32, name='dec_seq_len')
max_decoder_length = tf.reduce_max(pla_inp_len)

encoder_inputs = tf.one_hot(pl_inputs, en_vocab_size + 4)

ending = tf.strided_slice(pl_labels, [0, 0], [batch_size, -1], [1, 1])
dec_input = tf.concat([tf.fill([batch_size, 1], go), ending], 1)
decoder_inputs = tf.one_hot(dec_input, de_vocab_size + 4)

# Bidirectional layer uses bidirectional_dynamic_rnn
forward_cell = tf.contrib.rnn.LSTMCell(num_hidden/2)
backward_cell = tf.contrib.rnn.LSTMCell(num_hidden/2)
biout, bi_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, encoder_inputs,
                                                  dtype=tf.float32, time_major=False)
outputs_concat = tf.concat([biout[0], biout[1]], 2)

# Encoder takes the bidirectional output. Uses dynamic_rnn
encoder_cells = []
for _ in range(num_layers):
    cell = tf.contrib.rnn.LSTMCell(
        num_hidden, initializer=tf.random_uniform_initializer(-0.5, 0.5, seed=2))
    encoder_cells.append(cell)

encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, outputs_concat, dtype=tf.float32, time_major=False, scope="plapl_encoder")


def one_hot(inp):
    return tf.one_hot(inp, de_vocab_size + 4)


# Decoder also uses dynamic_rnn
def build_decoder(encoder_outputs, encoder_last_state, batch_size, helper, beam_size=1, reuse=None):
    with tf.variable_scope("seq2seq_decoder", reuse=reuse):
        if beam_size > 1:
            encoder_outputs = tf.contrib.seq2seq.tile_batch(
                encoder_outputs, multiplier=beam_size)
            encoder_last_state = tf.contrib.framework.nest.map_structure(
                lambda s: tf.contrib.seq2seq.tile_batch(s, beam_size), encoder_last_state)

        decoder_cells = [tf.contrib.rnn.LSTMCell(
            num_hidden, initializer=tf.random_uniform_initializer(-0.5, 0.5, seed=2))
            for i in range(num_layers)]
        decoder_initial_state = encoder_last_state
        initial_state = [state for state in encoder_last_state]

        if attention:
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=att_num_hidden, memory=encoder_outputs)
            decoder_cells[-1] = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cells[-1],
                attention_mechanism=attention_mechanism,
                attention_layer_size=att_num_hidden,
                name="attention",
                initial_cell_state=encoder_last_state[-1])
            initial_state[-1] = decoder_cells[-1].zero_state(
                batch_size=batch_size * beam_size, dtype=tf.float32)

        decoder_initial_state = tuple(initial_state)
        decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)

        output = tf.layers.Dense(de_vocab_size, activation=tf.nn.sigmoid,
                                 kernel_initializer=tf.truncated_normal_initializer(
                                    mean=0.0, stddev=0.1))

        if helper is None:
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=one_hot,
                start_tokens=tf.fill([batch_size], go),
                end_token=eos,
                length_penalty_weight=0.0,
                initial_state=decoder_initial_state,
                beam_width=beam_size,
                output_layer=output)
        else:
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, helper=helper, initial_state=decoder_initial_state,
                output_layer=output)

        decoder_train, final_state, final_seq_len = tf.contrib.seq2seq.dynamic_decode(
            decoder, output_time_major=False, maximum_iterations=de_seq_length)
    return decoder_train, final_state, final_seq_len


# Train Decoder
training_helper = tf.contrib.seq2seq.TrainingHelper(
    inputs=decoder_inputs, sequence_length=pla_lab_len, time_major=False, name='training_help')
train_out, final_state, final_seq_len = build_decoder(
    encoder_outputs, encoder_final_state, batch_size, training_helper, beam_size=1)

# Greedy Decoder
greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    embedding=one_hot, start_tokens=tf.fill([batch_size], go), end_token=eos)
greedy_out, final_infec, len_infer = build_decoder(
    encoder_outputs, encoder_final_state, batch_size, greedy_helper, beam_size=1, reuse=True)

# Beam Decoder
beam_out, final_infec, len_infer = build_decoder(
    encoder_outputs, encoder_final_state, batch_size, None, beam_size, reuse=True)

# Optimization
decoder_logits = tf.identity(train_out.rnn_output, 'logits')
final_logit = tf.log(tf.clip_by_value(decoder_logits, 1e-13, 1.0))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=pl_labels, logits=final_logit))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=momentum).minimize(loss)

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
            x, y, xl, yl = get_batch(batch_size, i)
            feed_dict = {pl_inputs: x, pl_labels: y, pla_inp_len: xl, pla_lab_len: yl}
            sess.run(train_op, feed_dict=feed_dict)
            # sess = tfdbg.LocalCLIDebugWrapperSession(sess)

            # Compute the average loss OPTIONAL
            avg_cost += sess.run(loss, feed_dict=feed_dict)/total_batch
            summary_str = sess.run(merged_summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, iteration*total_batch + 1)
        print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    saver.save(sess, 'savedmodels/model_num_seq2seq')
    print("Tuning Completed!")

    x, y, xl, yl = get_batch(batch_size, i)
    feed_dict = {pl_inputs: x, pl_labels: y, pla_inp_len: xl, pla_lab_len: yl}
    res_train, res_greedy, res_beam = sess.run([train_out, greedy_out, beam_out],
                                               feed_dict=feed_dict)

    print(y)
    print(res_train.sample_id)
    print(res_greedy.sample_id)
    print(res_beam.predicted_ids.transpose((2, 0, 1))[-1])
