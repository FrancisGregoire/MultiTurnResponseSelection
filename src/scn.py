import pickle

import numpy as np
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences

import utils

from evaluate import computeR10_1, computeR2_1


class SCN():
    def __init__(self):
        self.max_num_utterance = 10
        self.negative_samples = 1
        self.max_sentence_len = 50
        self.word_embedding_size = 200
        self.rnn_size = 200
        self.hidden_size = 50
        self.output_size = 2
        self.total_words = 434511
        self.batch_size = 40
        self.learning_rate = 0.001
        self.max_epoch = 10
        self.train_embeddings = False

    def LoadModel(self):
        #init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        #with tf.Session() as sess:
            #sess.run(init)
        saver.restore(sess,"neg5model\\model.5")
        return sess
        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.
        # with tf.Session() as sess:
        #     # Restore variables from disk.
        #     saver.restore(sess, "/model/model.5")
        #     print("Model restored.")

    def BuildModel(self):
        # Placeholders.
        self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len))  # We pad zeros if the number of utterances in a
        self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))                           # context is less than 10, otherwise we keep the
        self.y_true = tf.placeholder(tf.int32, shape=(None,))                                                      # last 10 utterances.
        self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance))
        self.response_len_ph = tf.placeholder(tf.int32, shape=(None,))
        self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))

        # Embeddings.
        word_embeddings = tf.get_variable(name='word_embeddings_v',
                                          shape=(self.total_words, self.word_embedding_size),
                                          dtype=tf.float32,
                                          trainable=self.train_embeddings)  # trainable is False by default???
        self.embedding_init = word_embeddings.assign(self.embedding_ph)
        # Utterance embeddings.
        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1)
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)
        # Response embeddings.
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)

        # GRU cells.
        sentence_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_size, kernel_initializer=tf.orthogonal_initializer())
        final_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_size, kernel_initializer=tf.orthogonal_initializer())

        # Encode response.
        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(cell=sentence_GRU,
                                                       inputs=response_embeddings,
                                                       sequence_length=self.response_len_ph,
                                                       dtype=tf.float32,
                                                       scope='sentence_GRU')
        self.response_embedding_save = response_GRU_embeddings

        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])  # Why transposing?
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])  # Why transposing?

        A_matrix = tf.get_variable(name='A_matrix_v',
                                   shape=(self.rnn_size, self.rnn_size),
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   dtype=tf.float32)

        # For each (utterance, response) pair.
        matching_vectors = []
        reuse = None
        for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
            # Compute M_1.
            matrix1 = tf.matmul(utterance_embeddings, response_embeddings)

            # Encode the utterance.
            utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(cell=sentence_GRU,
                                                            inputs=utterance_embeddings,
                                                            sequence_length=utterance_len,
                                                            dtype=tf.float32,
                                                            scope='sentence_GRU')

            # Compute M_2. (not sure to understand tf.einsum)
            matrix2 = tf.einsum('aij,jk->aik', utterance_GRU_embeddings, A_matrix)  # TODO:check this
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')

            # Convolution + pooling layers.
            conv_layer = tf.layers.conv2d(inputs=matrix,
                                          filters=8,
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding='VALID',  #'valid'?
                                          kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                          activation=tf.nn.relu,
                                          reuse=reuse,
                                          name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(inputs=conv_layer,
                                                    pool_size=(3, 3),
                                                    strides=(3, 3),
                                                    padding='VALID',  #'valid'?
                                                    name='max_pooling')  # TODO: check other params

            # Compute matching vector.
            matching_vector = tf.layers.dense(inputs=tf.contrib.layers.flatten(pooling_layer),
                                              units=self.hidden_size,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.tanh,
                                              reuse=reuse,
                                              name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector)

        # Encode matching vectors.
        _, last_hidden = tf.nn.dynamic_rnn(cell=final_GRU,
                                           inputs=tf.stack(matching_vectors, axis=0, name='matching_stack'),
                                           dtype=tf.float32,
                                           time_major=True,
                                           scope='final_GRU')  # TODO: check time_major

        # Linear output layer + softmax.
        logits = tf.layers.dense(inputs=last_hidden,
                                 units=self.output_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='final_v')
        self.y_pred = tf.nn.softmax(logits)

        # Loss + optimizer.
        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits))
        tf.summary.scalar('loss', self.total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)  # Add gradient clipping?
        self.train_op = optimizer.minimize(self.total_loss)

    def Evaluate(self, sess):
        with open(evaluate_file, 'rb') as f:
           utterances, last_utterance, labels = pickle.load(f)
        self.all_candidate_scores = []
        utterances, utterances_len = utils.multi_sequences_padding(utterances, self.max_num_utterance, self.max_sentence_len)
        utterances, utterances_len = np.array(utterances), np.array(utterances_len)
        last_utterance_len = np.array(utils.get_sequences_length(last_utterance, self.max_sentence_len))
        last_utterance = np.array(pad_sequences(last_utterance, padding='post', maxlen=self.max_sentence_len))
        low = 0
        while True:
            feed_dict = {self.utterance_ph: np.concatenate([utterances[low:low + 200]], axis=0),
                         self.all_utterance_len_ph: np.concatenate([utterances_len[low:low + 200]], axis=0),
                         self.response_ph: np.concatenate([last_utterance[low:low + 200]], axis=0),
                         self.response_len_ph: np.concatenate([last_utterance_len[low:low + 200]], axis=0),
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])
            low = low + 200
            if low >= utterances.shape[0]:
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)
        computeR10_1(all_candidate_scores,labels)
        computeR2_1(all_candidate_scores,labels)

    def TrainModel(self, continue_train=False, previous_modelpath="model"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('output', sess.graph)
            with open(embedding_file, 'rb') as f:
                embeddings = pickle.load(f, encoding="bytes")
            with open(utterance_file, 'rb') as f:
                utterances, last_utterance = pickle.load(f)
            with open(response_file, 'rb') as f:
                responses = pickle.load(f)
            utterances, utterances_len = utils.multi_sequences_padding(utterances, self.max_num_utterance, self.max_sentence_len)
            last_utterance_len = np.array(utils.get_sequences_length(last_utterance, self.max_sentence_len))
            last_utterance = np.array(pad_sequences(last_utterance, padding='post', maxlen=self.max_sentence_len))
            responses_len = np.array(utils.get_sequences_length(responses, self.max_sentence_len))
            responses = np.array(pad_sequences(responses, padding='post', maxlen=self.max_sentence_len))
            utterances, utterances_len = np.array(utterances), np.array(utterances_len)
            if continue_train == False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: embeddings})
            else:
                saver.restore(sess, previous_modelpath)
            low = 0
            epoch = 1
            while epoch < self.max_epoch:
                n_sample = min(low + self.batch_size, utterances.shape[0]) - low
                negative_indices = [np.random.randint(0, responses.shape[0], n_sample) for _ in range(self.negative_samples)]
                negs = [responses[negative_indices[i], :] for i in range(self.negative_samples)]
                negs_len = [responses_len[negative_indices[i]] for i in range(self.negative_samples)]
                feed_dict = {self.utterance_ph: np.concatenate([utterances[low:low + n_sample]] * (self.negative_samples + 1), axis=0),
                             self.all_utterance_len_ph: np.concatenate([utterances_len[low:low + n_sample]] * (self.negative_samples + 1), axis=0),
                             self.response_ph: np.concatenate([last_utterance[low:low + n_sample]] + negs, axis=0),
                             self.response_len_ph: np.concatenate([last_utterance_len[low:low + n_sample]] + negs_len, axis=0),
                             self.y_true: np.concatenate([np.ones(n_sample)] + [np.zeros(n_sample)] * self.negative_samples, axis=0)
                             }
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary)
                low += n_sample
                if low % 102400 == 0:
                    print("loss", sess.run(self.total_loss, feed_dict=feed_dict))
                    self.Evaluate(sess)
                if low >= utterances.shape[0]:
                    low = 0
                    saver.save(sess,"model/model.{0}".format(epoch))
                    print(sess.run(self.total_loss, feed_dict=feed_dict))
                    print('epoch={i}'.format(i=epoch))
                    epoch += 1


if __name__ == "__main__":
    embedding_file = "../data/Ubuntu/embedding.pkl"
    evaluate_file = "../data/Ubuntu/Evaluate.pkl"
    response_file = "../data/Ubuntu/responses.pkl"
    utterance_file = "../data/Ubuntu/utterances.pkl"
    max_num_utterance = 5
    max_sentence_len = 25

    scn = SCN()
    scn.BuildModel()
    scn.TrainModel()

    #sess = scn.LoadModel()
    #scn.Evaluate(sess)
    #print(len(results))

    # # Debugging.
    # init = tf.global_variables_initializer()
    # sess = tf.Session()

    # with open(embedding_file, 'rb') as f:
    #     embeddings = pickle.load(f, encoding="bytes")
    # with open(utterance_file, 'rb') as f:
    #     utterances, last_utterance = pickle.load(f)
    # with open(response_file, 'rb') as f:
    #     responses = pickle.load(f)

    # utterances, utterances_len = utils.multi_sequences_padding(utterances, max_num_utterance, max_sentence_len)
    # utterances, utterances_len = np.array(utterances), np.array(utterances_len)

    # last_utterance_len = np.array(utils.get_sequences_length(last_utterance, max_sentence_len))
    # last_utterance = np.array(pad_sequences(last_utterance, padding='post', maxlen=max_sentence_len))

    # responses_len = np.array(utils.get_sequences_length(responses, max_sentence_len))
    # responses = np.array(pad_sequences(responses, padding='post', maxlen=max_sentence_len))
