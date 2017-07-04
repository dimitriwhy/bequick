import numpy
import random
from segment_tree import SegmentTree

class Experience(object):
    def __init__(self, memory_size, batch_size, alpha):
        self.tree = SegmentTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha

    def add(self, data, priority):
        self.tree.add(data, priority**self.alpha)

    def select(self, beta):
        if self.tree.filled_size() < self.batch_size:
            return None, None, None

        out = []
        indices = []
        weights = []
        priorities = []
        #print(self.tree.cursor,self.tree.get_max_val())

        for _ in range(self.batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            #print(priority, self.tree.get_max_val())
            weights.append((1./self.memory_size/priority)**beta if priority > 1e-16 else 0)
            max_weight = max(weights)
            weights = [weight / max_weight for weight in weights]
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0]) # To avoid duplicating
        # self.priority_update(indices, priorities) # Revert priorities
        # print(weights)
        return out, weights, indices

    def priority_update(self, indices, priorities):
        for i, p in list(zip(indices, priorities)):
             self.tree.val_update(i, p**self.alpha)
    
    def reset_alpha(self, alpha):
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)

    def get_max_val(self) :
        max_val = self.tree.get_max_val()
        if max_val != 0 :
            return self.tree.get_max_val()**(-self.alpha)
        else :
            return 0
