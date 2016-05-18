import random

def json_byteify(input):
    """Converts a json object to byte-strings rather than unicode."""
    if isinstance(input, dict):
        return {json_byteify(key):json_byteify(value) for key,value in input.iteritems()}
    elif isinstance(input, list):
        return [json_byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

def mkdir_p(path):
    import os, errno
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def sample_weighted(weights, vals=None):
    """Selects a value from vals with probability proportional to the
    corresponding value in weights.

    If vals == None, returns the index that would have been picked
    """
	
    weightSum = sum(weights)
    if weightSum <= 0:
        if vals is None:
            return random.randint(0,len(weights)-1)
        return random.choice(vals)
    r = random.uniform(0.0,weightSum)
    if vals is None:
        for i,w in enumerate(weights):
            if r <= w:
                return i
            r -= w
        return len(weights)-1
    else:
        for v,w in zip(vals,weights):
            if r <= w:
                return v
            r -= w
        return vals[-1]
