import numpy as np

class PoseQuantizer:
    def __init__(self, min, max, count):
        self.min = min
        self.max = max
        self.count = count
        self.dx = (max - min) / count

    def encode_array(self, array):
        tmp = ((array - self.min) / self.dx).astype(np.int32)
        tmp[tmp < 0] = 0
        tmp[tmp > self.count - 1] = self.count - 1
        ids = [f"<extra_id_{i+1100}>" for i in tmp]
        return ids

    def decode_array(self, array):
        array = [str(i).replace( '<extra_id_', '') for i in array]
        array = [str(i).replace( '>', '') for i in array]
        array = [int(i)-1100 for i in array]
        array = np.asarray(array)

        tmp = (array * self.dx + self.min)
        tmp[tmp < self.min] = self.min
        tmp[tmp > self.max] = self.max
        return tmp