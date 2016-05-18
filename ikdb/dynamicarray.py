import numpy as np

def row_iterator(np_array):
    for item in xrange(np_array.shape[0]):
        yield np_array[item,:].tolist()
    return

class DynamicArray2D:
    """Utility class: allows a 2D array to be added to, then finalized into a
    numpy array using the compress() method.
    
    The decompress() method converts it back to a Python list.
    """
    def __init__(self,items=None):
        if items is None:
            items = []
        self.items = items
    def __iter__(self):
        try:
            return self.items.__iter__()
        except AttributeError:
            return row_iterator(self.array)
    def __len__(self):
        try:
            return len(self.items)
        except AttributeError:
            return self.array.shape[0]
    def __getitem__(self,item):
        try:
            return self.items[item]
        except AttributeError:
            return self.array[item,:].tolist()
    def __setitem__(self,item,value):
        try:
            self.items[item] = value
        except AttributeError:
            self.array[item,:] = value
    def __iadd__(self,array_like):
        if isinstance(array_like,np.ndarray):
            if hasattr(self,'array'):
                self.array = np.vstack((self.array,array_like))
            else:
                self.items += [array_like[row,:].tolist() for row in xrange(array_like.shape[0])]
        elif isinstance(array_like,(DynamicArray2D,list,tuple)):
            self.decompress()
            self.items += [array_like[i] for i in range(len(array_like))]
        return self
    def append(self,x):
        try:
            self.items.append(x)
        except AttributeError:
            self.decompress()
            self.items.append(x)
    def compress(self):
        try:
            self.array = np.array(self.items)
            del self.items
        except AttributeError:
            #already converted
            pass
    def decompress(self):
        try:
            n = self.array.shape[0]
            self.items = [self.array[item,:].tolist() for item in xrange(n)]
            del self.array
        except AttributeError:
            #already converted
            pass
