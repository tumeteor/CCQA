import tensorflow as tf
import logging


class candidate_answer_generation(object):
    def __init__(self, config, attention_layer, candidate_answers, sentence_num):
        self.config = config
        self.att_layer = attention_layer
        self.candidate_answers = candidate_answers
        self.sentence_num = sentence_num

    def get_candidate_answer_representations(self):
        # return answer: a correnct answer index
        # return predictions,
        candidate_answers = self.candidate_answers  # [sentence_num, candidate_number, constituency_idlist]
        sentence_candidate_answers = tf.gather(tf.gather(candidate_answers, 0), tf.range(
            tf.reduce_sum(tf.to_int32(tf.not_equal(tf.gather(candidate_answers, 0), -1)), [0, 1])))
        sentence_attentioned_hidden_states = tf.gather(self.att_layer.attentioned_hidden_states, 0)
        candidates_representations = self.get_candidates_representations_in_sentence(sentence_candidate_answers,
                                                                                     sentence_attentioned_hidden_states)
        # candidates_representations = tf.expand_dims(candidates_representations, 0)

        all_sentence_candidates_representations = tf.identity(candidates_representations)
        # sentence_num = tf.gather(tf.shape(self.att_layer.attentioned_hidden_states), 0)
        sentence_num = self.sentence_num
        logging.warning('attentioned_hidden_states:{}'.format(self.att_layer.attentioned_hidden_states))
        idx_var = tf.constant(1)

        def _recurse_sentence(sentences_candidates_representations, idx_var):
            # sentence_candidate_answers = tf.gather(tf.shape(candidate_answers), idx_var)
            sentence_candidate_answers = tf.gather(tf.gather(candidate_answers, idx_var), tf.range(
                tf.reduce_sum(tf.to_int32(tf.not_equal(tf.gather(candidate_answers, idx_var), -1)), [0, 1])))
            sentence_attentioned_hidden_states = tf.gather(self.att_layer.attentioned_hidden_states, idx_var)
            candidates_representations = self.get_candidates_representations_in_sentence(sentence_candidate_answers,
                                                                                         sentence_attentioned_hidden_states)
            # candidates_representations = tf.expand_dims(candidates_representations, 0)
            # sentences_candidates_representations = tf.concat([sentences_candidates_representations, candidates_representations],axis=0)
            sentences_candidates_representations = tf.concat(
                [sentences_candidates_representations, candidates_representations], axis=0)
            idx_var = tf.add(idx_var, 1)
            return sentences_candidates_representations, idx_var

        loop_cond = lambda a1, idx: tf.less(idx, sentence_num)
        loop_vars = [all_sentence_candidates_representations, idx_var]
        all_sentence_candidates_representations, idx_var = tf.while_loop(loop_cond, _recurse_sentence, loop_vars,
                                                                         shape_invariants=[tf.TensorShape(
                                                                             [None, 4 * self.config.hidden_dim]),
                                                                             idx_var.get_shape()])
        return all_sentence_candidates_representations

    def get_candidates_representations_in_sentence(self, sentence_candidate_answers,
                                                   sentence_attentioned_hidden_states):
        candidate_answer_num = tf.gather(tf.shape(sentence_candidate_answers), 0)
        logging.warning('candidate_answer_num:{}'.format(candidate_answer_num))
        logging.warning('sentence_candidate_answers:{}'.format(sentence_candidate_answers))
        candidate_answer_nodeids = tf.gather(sentence_candidate_answers, 0)
        # print(sentence_attentioned_hidden_states)# a node idx list
        # print(candidate_answer_nodeids)
        # print(tf.squeeze(candidate_answer_nodeids))
        candidate_answer_hidden_list = tf.gather(sentence_attentioned_hidden_states, candidate_answer_nodeids)
        # print(tf.shape(candidate_answer_hidden_list),candidate_answer_hidden_list)
        candidates_final_representations = self.get_candidate_answer_final_representations(candidate_answer_hidden_list)
        # print(candidate_final_representations)
        # candidates_final_representations = tf.expand_dims(candidate_final_representations, 0)
        idx_cand = tf.constant(1)

        # print(candidates_final_representations)

        def _recurse_candidate_answer(candidate_final_representations, idx_cand):
            cur_candidate_answer_nodeids = tf.gather(sentence_candidate_answers, idx_cand)
            # cur_candidate_answer_hidden_list = tf.gather(sentence_attentioned_hidden_states,
            #                                              cur_candidate_answer_nodeids)
            cur_candidate_answer_hidden_list = tf.gather(sentence_attentioned_hidden_states,
                                                         cur_candidate_answer_nodeids)
            # cur_candidate_final_representations = tf.expand_dims(
            #     self.get_candidate_answer_final_representations(cur_candidate_answer_hidden_list), 0)
            cur_candidate_final_representations = self.get_candidate_answer_final_representations(
                cur_candidate_answer_hidden_list)
            candidate_final_representations = tf.concat(
                [candidate_final_representations, cur_candidate_final_representations], axis=0)
            idx_cand = tf.add(idx_cand, 1)
            return candidate_final_representations, idx_cand

        loop_cond = lambda a1, idx: tf.less(idx, candidate_answer_num)
        loop_vars = [candidates_final_representations, idx_cand]
        candidates_final_representations, idx_cand = tf.while_loop(loop_cond, _recurse_candidate_answer, loop_vars,
                                                                   shape_invariants=[tf.TensorShape(
                                                                       [None, 4 * self.config.hidden_dim]),
                                                                       idx_cand.get_shape()])
        return candidates_final_representations

    def get_candidate_answer_final_representations(self, candidate_answer_hidden_list):
        output = tf.identity(candidate_answer_hidden_list)
        return output  # [2*hidden_dim] #[4*hidden_dim]
