from question_encoding_layer import *
from context_encoding_layer import *
import tensorflow as tf


class attentioned_layer(object):
    def __init__(self, config, q_encoding, c_encoding):
        input = q_encoding.bp_lstm.input
        self.num_leaves = q_encoding.bp_lstm.num_leaves
        hq = q_encoding.nodes_states
        hc = c_encoding.sentences_final_states
        self.inodes = q_encoding.bp_lstm.n_inodes
        constituent_num=self.num_leaves+self.inodes
        self.treestr = q_encoding.bp_lstm.treestr
        self.sentence_sum=c_encoding.c_bp_lstm.sentence_num
        self.c_inodes=c_encoding.c_bp_lstm.n_inodes
        self.c_leaves=c_encoding.c_bp_lstm.num_leaves
        self.hidden_dim=c_encoding.c_bp_lstm.hidden_dim
        self.attentioned_hidden_states=self.compute_state(hq,hc)

    def compute_state_hp(self,hq,hc,idx_batch):
        hp=tf.gather(hc,idx_batch)
        hp = tf.gather(hp, tf.range(
            tf.to_int32(tf.divide(tf.reduce_sum(tf.to_int32(tf.not_equal(hp, -1.0))), tf.constant(2*self.hidden_dim)))))

        inodes=tf.gather(self.c_inodes,idx_batch)
        num_leaves=tf.squeeze(tf.gather(self.c_leaves,idx_batch))
        n_nodes=tf.add(inodes,num_leaves)

        constituent_state=self.compute_state_hpc(hq,hp,0)
        idx_var=tf.constant(1)
        def _computestates(states,hq,hp,idx_var):
            cur_constituent_state=self.compute_state_hpc(hq,hp,idx_var)
            states=tf.concat([states,cur_constituent_state],axis=0)
            idx_var=tf.add(idx_var,1)
            return states,hq,hp,idx_var
        loop_cond=lambda a1,b1,c1,idx_var:tf.less(idx_var,n_nodes)
        loop_vars=[constituent_state,hq,hp,idx_var]
        states_h,hq,hp,idx_batch=tf.while_loop(loop_cond,_computestates, loop_vars, shape_invariants=[tf.TensorShape([None, 4*self.hidden_dim]),
                                                                         hq.get_shape(), hp.get_shape(), idx_var.get_shape()])
        states_h=tf.reshape(tf.pad(states_h,[[0,300-tf.gather(tf.shape(states_h),0)],[0,0]],"CONSTANT",constant_values=-1),[-1,4*self.hidden_dim])
        return tf.expand_dims(states_h,0)
    def compute_state_hpc(self,hq,hp,idx_batch):
        hpc=tf.gather(hp,idx_batch)
        hpc=tf.expand_dims(hpc,0)
        a = tf.matmul(hpc,hq,transpose_b=True)
        al = tf.identity(tf.squeeze(a))
        treestr=tf.gather(self.treestr,tf.range(self.inodes))
        b_leaves, a_leaves = self.compute_leaves(al, self.num_leaves, hq, hpc)
        b_nodes=tf.identity(b_leaves)
        a_nodes=tf.identity(a_leaves)
        idx_var=tf.constant(0)
        def reccurence(b_nodes,a_nodes,idx_var):
            node_info=tf.gather(treestr,idx_var)
            child_b=tf.gather(b_nodes,node_info)
            child_a=tf.gather(a_nodes,node_info)
            al_node = tf.divide(tf.exp(tf.gather(al,idx_var+self.num_leaves)),tf.reduce_sum(tf.map_fn(lambda x:tf.exp(x),child_a),axis=0,keep_dims=True))
            bl_node = tf.multiply(al_node,tf.add(hpc,tf.matmul(tf.expand_dims(child_a,0),child_b)))
            b_nodes = tf.concat(axis=0,values=[b_nodes,bl_node])
            a_nodes = tf.concat(axis=0,values=[a_nodes,al_node])
            idx_var = tf.add(idx_var,1)
            return b_nodes,a_nodes,idx_var
        loop_cond = lambda a1,b1,idx_var: tf.less(idx_var,self.inodes)
        loop_vars = [b_nodes,a_nodes,idx_var]
        b_nodes,a_nodes,idx_var = tf.while_loop(loop_cond,reccurence,loop_vars)
        br = tf.gather(b_nodes,self.num_leaves+self.inodes-1)
        hpc=tf.squeeze(hpc)
        hpc=tf.concat(axis=0,values=[hpc,br])
        hpc=tf.expand_dims(hpc,0)
        return hpc
    def compute_state(self,hq,hc,idx_batch=0):
        states_h=self.compute_state_hp(hq,hc,0)
        idx_batch=tf.constant(1)
        def _computestates(states,hq,hc,idx_batch):
            cur_states=self.compute_state_hp(hq,hc,idx_batch)
            states=tf.concat([states,cur_states],axis=0)
            idx_batch=tf.add(idx_batch,1)
            return states,hq,hc,idx_batch
        loop_cond=lambda a1,b1,c1,idx_var:tf.less(idx_var,self.sentence_sum)
        loop_vars=[states_h,hq,hc,idx_batch]
        states_h,hq,hc,idx_batch=tf.while_loop(loop_cond,_computestates, loop_vars, back_prop=False, shape_invariants=[tf.TensorShape([None, None, 4*self.hidden_dim]),
                                                                         hq.get_shape(), hc.get_shape(), idx_batch.get_shape()])
        return states_h
    def compute_leaves(self,al,num_leaves,hq,hpc):
        a_leaves=tf.gather(al,tf.range(num_leaves))
        a_leaves=tf.expand_dims(a_leaves,0)
        b=tf.matmul(a_leaves,hpc,transpose_a=True,transpose_b=False)
        b_leaves=tf.gather(b,tf.range(num_leaves))
        a_leaves=tf.squeeze(a_leaves)
        return b_leaves,a_leaves









