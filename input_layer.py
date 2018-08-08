import numpy as np
import logging
from tf_treenode import tNode, processTree
from collections import deque


class input_layer(object):
    def __init__(self, config, data):
        self.config = config
        self.question_data = data[0]
        self.answer_data = data[1]
        self.context_data = data[2]
        self.parsed_data = self.parse_data()
        self.sentence_num = len(self.context_data)

    def parse_data(self):
        b_input, b_treestr, t_input, t_treestr, t_parent = self.extract_filled_tree(self.question_data)
        c_inputs, c_treestrs, c_t_inputs, c_t_treestrs, c_t_parents = [], [], [], [], []
        for i in range(len(self.context_data)):
            c_input, c_treestr, c_t_input, c_t_treestr, c_t_parent = self.extract_filled_tree(self.context_data[i])
            c_inputs.append(c_input)
            c_treestrs.append(c_treestr)
            c_t_inputs.append(c_t_input)
            c_t_treestrs.append(c_t_treestr)
            c_t_parents.append(c_t_parent)
        return b_input, b_treestr, t_input, t_treestr, t_parent, c_inputs, c_treestrs, c_t_inputs, c_t_treestrs, c_t_parents

    def extract_filled_tree(self, cur_data):
        # cur_data is a treeroot
        dim2 = self.config.maxnodesize
        # dim1: batch_size
        # dim2: tree node size
        leaf_emb_arr = np.empty([dim2], dtype='int32')
        leaf_emb_arr.fill(-1)
        treestr_arr = np.empty([dim2, 2], dtype='int32')
        treestr_arr.fill(-1)
        t_leaf_emb_arr = np.empty([dim2], dtype='int32')
        t_leaf_emb_arr.fill(-1)
        t_treestr_arr = np.empty([dim2], dtype='int32')
        t_treestr_arr.fill(-1)
        t_par_leaf_arr = np.empty([dim2], dtype='int32')
        t_par_leaf_arr.fill(-1)
        tree = cur_data
        input_, treestr, t_input, t_treestr, t_par_leaf = self.extract_tree_data(tree, max_degree=2,
                                                                                 only_leaves_have_vals=False)
        leaf_emb_arr[0:len(input_)] = input_
        treestr_arr[0:len(treestr), 0:2] = treestr
        t_leaf_emb_arr[0:len(t_input)] = t_input
        t_treestr_arr[0:len(t_treestr)] = t_treestr
        t_par_leaf_arr[0:len(t_par_leaf)] = t_par_leaf
        return leaf_emb_arr, treestr_arr, t_leaf_emb_arr, t_treestr_arr, t_par_leaf_arr

    def extract_tree_data(self, tree, max_degree=2, only_leaves_have_vals=True):
        leaves, inodes = self.BFStree(tree)
        leaf_emb = []
        tree_str = []
        t_leaf_emb = []
        t_tree_str = []
        i = 0
        for leaf in reversed(leaves):
            leaf.idx = i
            i += 1
            leaf_emb.append(leaf.word)
        for node in reversed(inodes):
            node.idx = i
            c = [child.idx for child in node.children]
            tree_str.append(c)
            i += 1
        i = 0
        for node in inodes:
            node.tidx = i
            i += 1
            if node.parent:
                t_tree_str.append(node.parent.tidx)
            else:
                t_tree_str.append(-1)
        t_par_leaf = []
        for leaf in leaves:
            leaf.tidx = i
            i += 1
            t_par_leaf.append(leaf.parent.tidx)
            t_leaf_emb.append(leaf.word)
        print('{}:leaf'.format(leaf_emb))
        print('{}.tree'.format(tree_str))
        print('{}.t_tree'.format(t_tree_str))
        print('{}.t_leaf'.format(t_leaf_emb))
        print('{}.t_par_leaf'.format(t_par_leaf))
        return (
            np.array(leaf_emb, dtype='int32'), np.array(tree_str, dtype='int32'), np.array(t_leaf_emb, dtype='int32'),
            np.array(t_tree_str, dtype='int32'), np.array(t_par_leaf, dtype='int32'))

    def BFStree(self, root):
        node = root
        leaves = []
        inodes = []
        queue = deque([node])
        func = lambda node: node.children == []
        while queue:
            node = queue.popleft()
            if func(node):
                if node.word in self.config.word2idx:
                    node.word = self.config.word2idx[node.word]
                elif str(node.word).lower() in self.config.word2idx:
                    node.word = self.config.word2idx[str(node.word).lower()]
                else:
                    logging.warning('no word2idx for question/context {}'.format(node.word))
                    node.word = self.config.word2idx['unknown']
                leaves.append(node)
            else:
                inodes.append(node)
            if node.children:
                for child in node.children:
                    child.add_parent(node)
                queue.extend(node.children)
        return leaves, inodes
