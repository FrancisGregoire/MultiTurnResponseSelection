import concurrent.futures
import pickle

import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence


def build_data(lines, word_dict, tid=0):
    def word2id(c):
        if c in word_dict:
            return word_dict[c]
        else:
            return 0

    cnt = 0
    utterances = []
    last_utterance = []
    for line in lines:
        fields = line.rstrip().lower().split('\t')
        utterance = fields[1].split('###')
        utterances.append([list(map(word2id, text_to_word_sequence(each_utt))) for each_utt in utterance])
        last_utterance.append(list(map(word2id, text_to_word_sequence(fields[2]))))
        cnt += 1
        if cnt % 10000 == 0:
            print(tid, cnt)
    return utterances, last_utterance


def build_evaluate_data(lines, tid=0):
    with open('worddata/word_dict.pkl', 'rb') as f:
        word_dict = pickle.load(f)

    def word2id(c):
        if c in word_dict:
            return word_dict[c]
        else:
            return 0

    cnt = 0
    utterances = []
    last_utterance = []
    for line in lines:
        fields = line.rstrip().lower().split('\t')
        utterance = fields[-1].split('###')
        utterances.append([list(map(word2id, text_to_word_sequence(each_utt))) for each_utt in utterance])
        last_utterance.append(list(map(word2id, text_to_word_sequence(fields[0]))))
        cnt += 1
        if cnt % 10000 == 0:
            print(tid, cnt)
    return utterances, last_utterance


def multi_sequences_padding(all_sequences, max_num_utterance=10, max_sentence_len=50):
    PAD_SEQUENCE = [0] * max_sentence_len
    padded_sequences = []
    sequences_length = []
    for sequences in all_sequences:
        sequences_len = len(sequences)
        sequences_length.append(get_sequences_length(sequences, max_sentence_len))
        if sequences_len < max_num_utterance:
            sequences += [PAD_SEQUENCE] * (max_num_utterance - sequences_len)
            sequences_length[-1] += [0] * (max_num_utterance - sequences_len)
        else:
            sequences = sequences[-max_num_utterance:]
            sequences_length[-1] = sequences_length[-1][-max_num_utterance:]
        sequences = pad_sequences(sequences, padding='post', maxlen=max_sentence_len)
        padded_sequences.append(sequences)
    return padded_sequences, sequences_length


def get_sequences_length(sequences, max_sentence_len):
    sequences_length = [min(len(sequence), max_sentence_len) for sequence in sequences]
    return sequences_length


def load_data(total_words):
    process_num = 10
    executor = concurrent.futures.ProcessPoolExecutor(process_num)
    base = 0
    results = []
    utterances = []
    last_utterance = []
    word_dict = dict()
    vectors = []
    with open('data/glove.twitter.27B.200d.txt', encoding='utf8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            word_dict[line[0]] = i
            vectors.append(line[1:])
            if i > total_words:
                break
    with open('worddata/embedding_matrix.pkl', "wb") as f:
        pickle.dump(vectors, f)
    with open("data/biglearn_train.old.txt", encoding="utf8") as f:
        lines = f.readlines()
        total_num = 1000000
        print(total_num)
        low = 0
        step = total_num // process_num
        print(step)
        while True:
            if low < total_num:
                results.append(executor.submit(build_data, lines[low:low + step], word_dict, base))
            else:
                break
            base += 1
            low += step

        for result in results:
            h, t = result.result()
            utterances += h
            last_utterance += t
    print(len(utterances))
    print(len(last_utterance))
    pickle.dump([utterances, last_utterance], open("worddata/train.pkl", "wb"))
    responses_id = []
    with open('emb/responses.txt', encoding='utf8') as f:
        responses = f.readlines()

    def word2id(c):
        if c in word_dict:
            return word_dict[c]
        else:
            return 0

    for action in responses:
        responses_id.append([word2id(word) for word in text_to_word_sequence(action)])
    with open('worddata/responses_embeddings.pkl', 'wb') as f:
        pickle.dump(responses_id, f)


def evaluate(test_file, sess, responses, responses_len, max_sentence_len, utterance_ph, all_utterance_len_ph,
             response_ph, response_len, y_pred):
    each_test_run = len(responses) // 3
    acc1 = [0.0] * 10
    rank1 = 0.0
    cnt = 0
    print('evaluating')

    with open(test_file, encoding="utf8") as f:
        lines = f.readlines()
        low = 0
        utterances, last_utterance = build_evaluate_data(lines)
        utterances, utterances_len = multi_sequences_padding(utterances, max_num_utterance=10, max_sentence_len=max_sentence_len)
        last_utterance_len = np.array(get_sequences_length(last_utterance, max_sentence_len))
        last_utterance = np.array(pad_sequences(last_utterance,padding='post', maxlen=max_sentence_len))
        utterances, utterances_len = np.array(utterances), np.array(utterances_len)
        feed_dict = {utterance_ph: utterances,
                     all_utterance_len_ph: utterances_len,
                     response_ph: last_utterance,
                     response_len: last_utterance_len
                     }
        true_scores = sess.run(y_pred, feed_dict=feed_dict)
        true_scores = true_scores[:, 1]
        for i in range(true_scores.shape[0]):
            all_candidate_scores = []
            for j in range(3):
                feed_dict = {utterance_ph: np.concatenate([utterances[low:low + 1]] * each_test_run, axis=0),
                             all_utterance_len_ph: np.concatenate([utterances_len[low:low + 1]] * each_test_run, axis=0),
                             response_ph: responses[each_test_run * j:each_test_run * (j + 1)],
                             response_len: responses_len[each_test_run * j:each_test_run * (j + 1)]
                             }
                candidate_scores = sess.run(y_pred, feed_dict=feed_dict)
                all_candidate_scores.append(candidate_scores[:, 1])
            all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
            pos1 = np.sum(true_scores[i] + 1e-8 < all_candidate_scores)
            if pos1 < 10:
                acc1[pos1] += 1
            rank1 += pos1
            low += 1
        cnt += true_scores.shape[0]
    print([a / cnt for a in acc1])  # rank top 1 to top 10 acc
    print(rank1 / cnt)  # average rank
    print(np.sum(acc1[:3]) * 1.0 / cnt)  # top 3 acc


if __name__ == '__main__':
    load_data(10)
