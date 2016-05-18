import itertools 

def Min():
    minval = None
    while True:
        i = yield
        if minval is None:
            minval = i
        else:
            minval = min(i,minval)
        yield minval

def Max():
    maxval = None
    while True:
        i = yield
        if maxval is None:
            maxval = i
        else:
            maxval = max(i,maxval)
        yield maxval

def Count():
    cnt = 0
    while True:
        i = yield
        cnt += 1
        yield cnt

def Sum(startval=0):
    s = startval
    while True:
        i = yield
        s += i
        yield s

def Product(startval=1):
    s = startval
    while True:
        i = yield
        s *= i
        yield s

def Series(function,startval=0):
    s = startval
    while True:
        i = yield
        s += function(i)
        yield s

def SumSquared(startval=0):
    return Series(lambda x:x*x,startval)

def Mean():
    s = 0
    c = 0
    while True:
        i = yield
        s += i
        c += 1
        yield float(s)/c

def Histogram(bucketSize=None,maxBuckets=None):
    from collections import defaultdict
    buckets = defaultdict(int)
    while True:
        i = yield
        if maxBuckets and len(buckets) >= maxBuckets:
            yield buckets
        else:
            if bucketSize is not None:
                buckets[int(float(i)/bucketSize)] += 1
            else:
                buckets[i] += 1
        yield buckets

def Variance():
    s = 0
    s2 = 0
    c = 0
    while True:
        i = yield
        s2 += i*i
        s += i
        c += 1
        yield float(s2)/c - (float(s)/c)**2

def StdDeviation():
    import math
    var = Variance()
    while True:
        x = yield
        var.next()
        v = var.send(x)
        yield math.sqrt(v)

def WeightedSum(startval=0):
    s = startval
    while True:
        pair = yield
        if hasattr(pair,'__iter__'):
            i,w = pair
            s += i*w
        else:
            s += pair
        yield s

def WeightedMean(startval=0):
    s = startval
    wsum = 0
    while True:
        pair = yield
        if hasattr(pair,'__iter__'):
            i,w = pair
            s += i*w
            wsum += w
        else:
            s += pair
            wsum += 1
        yield float(s)/wsum

def Multiple(*funcs):
    while True:
        x = yield
        vals = []
        for f in funcs:
            f.next()
            vals.append(f.send(x))
        yield vals

class Accumulator:
    def __init__(self,generator,label=None):
        if hasattr(generator,'__len__'):
            self.generator = Multiple(*generator)
        else:
            self.generator = generator
        self.value = None
        self.label = label
    def add(self,x):
        self.generator.next()
        self.value = self.generator.send(x)
    def asdict(self):
        if self.value is None:
            return dict()
        elif self.label is not None:
            if hasattr(self.value,'__iter__'): #multiple
                return dict(zip(self.label,self.value))
            else:
                return {self.label:self.value}
        else:
            if hasattr(self.value,'__iter__'): #multiple
                return dict(zip(range(len(self.value)),self.value))
            else:
                return {'value':self.value}

if __name__=="__main__":
    m = Accumulator([Min(),Max(),WeightedMean()],['min','max','weighted_mean'])
    m.add(1)
    m.add(2)
    m.add(3)
    m.add(4)
    print m.asdict()
