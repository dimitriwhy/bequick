import sys
import os
import math
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('chen2014-rl')

class SegmentTree(object) :
    def __init__(self, max_size) :
        self.max_size = int(max_size)
        self.tree_size = int(max_size)
        self.tree = [0 for i in range(4 * self.tree_size)]
        self.max_tree = [0 for i in range(4 * self.tree_size)]
        self.data = [None for i in range(4 * self.max_size)]
        self.size = 0
        self.cursor = 0
        
    def _add(self, root, left, right, position, contents, value) :
        if left == right :
            # contents == None for update value
            if contents != None :
                self.data[root] = contents
            self.tree[root] = value
            self.max_tree[root] = value
            return
        middle = (left + right) // 2
        if position <= middle :
            self._add(root * 2, left, middle, position, contents, value)
        else :
            self._add(root * 2 + 1, middle + 1, right, position, contents, value)

        self.tree[root] = self.tree[root * 2] + self.tree[root * 2 + 1]

        self.max_tree[root] = max(self.max_tree[root * 2], self.max_tree[root * 2 + 1])
        return
    
    def add(self, contents, value) :
        index = self.cursor
        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self._add(1, 0, self.max_size - 1, index, contents, value)

    def _find(self, root, left, right, value) :
        if left == right :
            return self.data[root], self.tree[root] / self.tree[1], left
        middle = (left + right) // 2
        if value <= self.tree[root * 2] :
            return self._find(root * 2, left, middle, value)
        else :
            return self._find(root * 2 + 1, middle + 1, right, value - self.tree[root * 2])
        
    def find(self, value, norm = True) :
        if norm:
            #LOG.info("self.tree[1]={0}".format(self.tree[1]))
            value *= self.tree[1]
        return self._find(1, 0, self.max_size - 1, value)

    def val_update(self, index, value) :
        self._add(1, 0, self.max_size - 1, index, None, value)
        return

    def _get_val(self, root, left, right, position) :
        if left == right :
            return self.tree[root]
        middle = (left + right) // 2
        if position <= left :
            return self._get_val(root * 2, left, middle, position)
        else :
            return self._get_val(root * 2 + 1, middle + 1, right, position)

    def get_val(self, index) :
        return self._get_val(1, 0, self.max_size - 1, index)

    def get_max_val(self) :
        return self.max_tree[1]

    def filled_size(self) :
        return self.size
