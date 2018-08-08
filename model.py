from tensorflow.python import debug as tf_debug
import traceback
from attention_layer import attentioned_layer
from question_encoding_layer import question_encoding
from context_encoding_layer import context_encoding
from input_layer import input_layer
from candidate_answer_generation_layer import candidate_answer_generation
from prediction_layer import predict
import logging
from collections import deque
import numpy as np
import tensorflow as tf
import time


class ccrc_model(object):
    def __init__(self, config):
        self.config = config

    def train(self, data, restore):
        q_encoding = question_encoding(self.config)
        c_encoding = context_encoding(self.config)
        attention_layer = attentioned_layer(self.config, q_encoding, c_encoding)

        correct_answer_idx = tf.placeholder(tf.int32, name='correct_answer_index')
        candidate_answers = tf.placeholder(tf.int32, [None, None, None], name='candidate_answers')

        candidate_answers_with_states = candidate_answer_generation(self.config, attention_layer, candidate_answers,
                                                                    c_encoding.c_bp_lstm.sentence_num) \
            .get_candidate_answer_representations()
        candidate_answers_with_states = tf.reshape(candidate_answers_with_states, [-1, 2 * self.config.maxnodesize])

        with tf.variable_scope('projection_layer'):
            softmax_W = tf.get_variable('softmax_w', [4 * self.config.hidden_dim, 1],
                                        initializer=tf.random_normal_initializer(mean=0,
                                                                                 stddev=1 / self.config.hidden_dim))
            softmax_b = tf.get_variable('softmax_b', [1], initializer=tf.constant_initializer(0.0))

        answer_prediction = predict(candidate_answers_with_states)
        correct_answer = tf.one_hot(correct_answer_idx, tf.gather(tf.shape(candidate_answers_with_states), 0))
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.expand_dims(answer_prediction, 0),
                                                          labels=tf.expand_dims(correct_answer, 0))
        loss = tf.reshape(tf.squeeze(loss),[])
        global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('global_step',global_step)
        merged = tf.summary.merge_all()

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=self.config.lr,
            optimizer='Adagrad'
        )
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            if restore:
                saver.restore(sess, "./ckpt/tree_rnn_weights")
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
                init_l = tf.local_variables_initializer()
                sess.run(init_l)

            #coord = tf.train.Coordinator()
            #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter("logs/", graph=tf.get_default_graph())
            for epoch in range(20):
                losses = []
                
                logging.warn(len(data))
                for curidx in range(len(data)):
                    try:
                        start = time.time()
                        inputs = input_layer(self.config, data[curidx])
                        cur_candidate_answers, cur_answer_idx, _ = self.load_candidate_answer(inputs.answer_data,
                                                                                              inputs.context_data)

                        b_input, b_treestr, t_input, t_treestr, t_parent, \
                        c_inputs, c_treestrs, c_t_inputs, c_t_treestrs, c_t_parents = inputs.parsed_data
                        feed = {
                            q_encoding.bp_lstm.input: b_input,
                            q_encoding.bp_lstm.treestr: b_treestr,
                            q_encoding.td_lstm.t_input: t_input,
                            q_encoding.td_lstm.t_treestr: t_treestr,
                            q_encoding.td_lstm.t_par_leaf: t_parent,

                            c_encoding.c_bp_lstm.input: c_inputs,
                            c_encoding.c_bp_lstm.sentence_num: inputs.sentence_num,
                            c_encoding.c_bp_lstm.treestr: c_treestrs,
                            c_encoding.c_td_lstm.t_input: c_t_inputs,
                            c_encoding.c_td_lstm.t_treestr: c_t_treestrs,
                            c_encoding.c_td_lstm.t_par_leaf: c_t_parents,

                            correct_answer_idx: cur_answer_idx,
                            candidate_answers: cur_candidate_answers
                        }
                        _, curloss, summary = sess.run([train_op, loss, merged], feed_dict=feed)
                        logging.warn(tf.train.global_step(sess,tf.train.get_global_step()))
                        summary_writer.add_summary(summary, global_step=curidx)
                        losses.append(curloss)
                        logging.warn(
                            "{}th training, loss={}, avg loss in this epoch={}, cost time={}".format(curidx, curloss,
                                                                                                     sum(losses) / len(
                                                                                                         losses),
                                                                                                     time.time() - start))
                    except:
                        logging.warn("session error:") 
                        traceback.print_exc()
                logging.warn("{}th epoch, avg loss={}".format(epoch, sum(losses)/len(losses)))
            saver.save(sess, "./ckpt/tree_rnn_weights")

    def load_candidate_answer(self, answer_data, context_sentence_roots_list):
        # candidate_answers: sentence_num * candidate_number * constituency_num, each is a constituency id list(reversed BFS order)
        # correct_answer_idx
        candidate_answers = []
        correct_answer_idx = -1
        candidate_answer_overall_number = 0
        dim2 = self.config.maxnodesize
        sentence_num = len(context_sentence_roots_list)
        overall_idx = -1
        for root in context_sentence_roots_list:
            overall_idx += 1
            cur_candidate_answer_final = np.empty([dim2, 1], dtype='int32')
            cur_candidate_answer_final.fill(-1)
            cur_candidate_answer = []
            constituency_id2span = {}
            leaf_num = 0
            node = root  #############
            queue = deque([node])
            while queue:
                node = queue.popleft()
                if node.children != []:
                    candidate_answer_overall_number += 1
                    cur_candidate_answer.append([node.idx])
                    overall_idx += 1
                    queue.extend(node.children)
                    constituency_id2span[node.idx] = node.span
                    # print("node.span",node.span)
                    # print("getspans",node.get_spans)
                    # print("ans",answer_data)
                    if node.span == answer_data:
                        if correct_answer_idx != -1:
                            logging.warning('{} has duplicated candidate answers'.format(root.span))
                            correct_answer_idx = overall_idx
                        else:
                            correct_answer_idx = overall_idx
            if cur_candidate_answer != []:
                cur_candidate_answer_final[0:len(cur_candidate_answer)] = cur_candidate_answer
            candidate_answers.append(cur_candidate_answer_final)
        # print("correct answer",correct_answer_idx,candidate_answer_overall_number)
        # print(1)
        return candidate_answers, correct_answer_idx, candidate_answer_overall_number
