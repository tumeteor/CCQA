import tensorflow as tf


def predict(predictions):
    with tf.variable_scope('projection_layer', reuse=True):
        softmax_w = tf.get_variable('softmax_w')
        softmax_b = tf.get_variable('softmax_b')
        scores = tf.matmul(predictions, softmax_w) + softmax_b
        scores = tf.squeeze(scores)  # [candidate_answer_num]
        return scores
