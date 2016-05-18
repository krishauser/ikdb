"""Module for generating and testing feature mappings of hierarchical objects.

Some terminology:
- A hierarchical object is a class or a potentially nested collection of dicts,
  lists (or tuples), and primitive objects (usually bool,int,float,str although
  other objects may be supported).
- A feature path in a hierarchical object is key or a list of keys of the
  collection.  These keys may only be strings and integers.
- A feature mapping is a list of feature paths, each of which maps to a node in
  the hierarchical object.
- A feature vector is a flattened list of primitive objects that is extracted
  from a hierarchical object, given a feature mapping.  It can also be injected
  into a hierarchical object of the same structure, given the same feature
  mapping.
"""
from collections import defaultdict
import accumulator

def _extract_one(object,feature):
    if isinstance(feature,(list,tuple)):
        if len(feature) > 1:
            #nested feature
            return _extract_one(_extract_one(object,feature[0]),feature[1:])
        else:
            #base case of nesting
            feature = feature[0]
    if isinstance(feature,int):
        return object[feature]
    elif isinstance(feature,str) and hasattr(object,feature):
        return object.feature
    else:
        return object[feature]

def _flatten(object):
    """Given a hierarchical object of classes, lists, tuples, dicts, or primitive
    values, flattens all of the values in object into a single list.
    """
    if isinstance(object,(list,tuple)):
        return sum([_flatten(v) for v in object],[])
    elif isinstance(object,dict):
        return sum([_flatten(v) for v in object.itervalues()],[])
    else:
        return [object]

def extract(object,features):
    """Given a hierarchical object 'object' and a list of feature paths
    'features', returns the values of object indexed by those feature paths
    as a flattened list.
    
    Note: a path in features may reference an internal node,
    in which case the return result will contain all values under that
    internal node.

    A simple example:
        object = {'name':'Joe','account':1234,'orders':[2345,3456]}
        features = ['account','orders']
        extract(object,features) => [1234,2345,3456]

    A more complex example:
        features = ['account',['orders',0],['orders',1]]
        extract(object,features) => [1234,2345,3456]

    Note: feature paths may only be strings, integers, or lists of strings
    and integers.
    """
    v = []
    for f in features:
        try:
            v.append(_extract_one(object,f))
        except Exception:
            print "Error extracting feature",f,"from",object
            raise
    return _flatten(v)

def _fill(object,valueIter):
    if isinstance(object,(list,tuple)):
        for i in xrange(len(object)):
            if hasattr(object[i],'__iter__'):
                _fill(object[i],valueIter)
            else:
                object[i] = valueIter.next()
    elif isinstance(object,dict):
        for i in object:
            if hasattr(object[i],'__iter__'):
                _fill(object[i],valueIter)
            else:
                object[i] = valueIter.next()
    else:
        raise RuntimeError("_fill can only be called with a container type")

def _inject_one(object,feature,valueIter):
    if isinstance(feature,(list,tuple)):
        if len(feature) > 1:
            _inject_one(_extract_one(object,feature[0]),feature[1:],valueIter)
            return
        else:
            feature = feature[0]
    if isinstance(feature,int):
        if hasattr(object[feature],'__iter__'):
            _fill(object[feature],valueIter)
        else:
            object[feature]=valueIter.next()
    elif hasattr(object,feature):
        if hasattr(object.feature,'__iter__'):
            _fill(object.feature,valueIter)
        else:
            object.feature=valueIter.next()
    else:
        if hasattr(object[feature],'__iter__'):
            _fill(object[feature],valueIter)
        else:
            object[feature]=valueIter.next()

def inject(object,features,values):
    """Given a hierarchical structure 'object', a list of feature paths
    'features', and a list of values 'values',
    sets those values of object indexed by those feature names
    to the corresponding entries in 'values'.
    
    Note: the feature paths may reference internal nodes, in which case the
    internal nodes are extracted

    A simple example:
        object = {'name':'Joe','account':1234,'orders':[2345,3456]}
        features = ['account','orders']
        inject(object,features,[1235,2346,3457]]) => object
          now contains {'name':'Joe','account':1235,'orders':[2346,3457]}

    A more complex example:
        features = [['orders',1]]
        inject(object,features,[3458]) => object now contains
          {'name':'Joe','account':1235,'orders':[2346,3458]}

    Note: features may only be strings, integers, or lists of strings
    and integers.
    """
    viter = iter(values)
    for f in features:
        _inject_one(object,f,viter)


def structure(object,hashable=True):
    """Returns an object describing the hierarchical structure of the given
    object (eliminating the values).  Structures can be then compared via
    equality testing.  This can be used to more quickly compare
    two structures than structureMatch, particularly when hashable=True.

    If hashable = True, this returns a hashable representation.  Otherwise,
    it returns a more human-readable representation.
    """
    if isinstance(object,(list,tuple)):
        res= [structure(v) for v in object]
        if all(v is None for v in res):
            #return a raw number
            return len(res)
        if hashable: return tuple(res)
        return res
    elif isinstance(object,dict):
        res = dict()
        for k,v in object.iteritems():
            res[k] = structure(v)
        if hashable: return tuple(res.items())
        return res
    else:
        return None

def structureMatch(object1,object2):
    """Returns true if the objects have the same hierarchical structure
    (but not necessarily the same values)."""
    if isinstance(object1,(list,tuple)):
        if not isinstance(object2,(list,tuple)): return False
        if len(object1) != len(object2): return False
        for (a,b) in zip(object1,object2):
            if not structureMatch(a,b): return False
        return True
    elif isinstance(object1,dict):
        if not isinstance(object2,dict): return False
        if len(object1) != len(object2): return False
        for k,v in object1.iteritems():
            try:
                v2 = object2[k]
            except KeyError:
                return False
            if not structureMatch(v,v2): return False
        return True
    if hasattr(object1,'__iter__'):
        if not hasattr(object2,'__iter__'):
            return False
        #TODO: check other collections?
        return True
    else:
        if hasattr(object2,'__iter__'):
            return False
        #TODO: check for compatibility between classes?
        return True

def schema(object):
    """Returns an object describing the hierarchical structure of the given
    object, with Nones in place of the values.  During schemaMatch, None's
    match with any value.  The None values can also be replaced with values
    to enforce specific value matches, or boolean predicates to enforce
    more general matches.
    """
    if isinstance(object,(list,tuple)):
        res= [schema(v) for v in object]
        return res
    elif isinstance(object,dict):
        res = dict()
        for k,v in object.iteritems():
            res[k] = structure(v)
        return res
    else:
        return None

def schemaMatch(schema,object):
    """Returns true if the object matches the given schema."""
    if schema is None: return True
    if isinstance(schema,(list,tuple)):
        if not isinstance(object,(list,tuple)): return False
        if len(schema) != len(schema): return False
        for (a,b) in zip(schema,object):
            if not schemaMatch(a,b): return False
        return True
    elif isinstance(schema,dict):
        if not isinstance(schema,dict): return False
        if len(schema) != len(object): return False
        for k,v in schema.iteritems():
            try:
                v2 = object[k]
            except KeyError:
                return False
            if not schemaMatch(v,v2): return False
        return True
    elif hasattr(schema, '__call__'): #predicate
        return schema(object)
    else:
        return (schema==object)

class HierarchicalAccumulator:
    """A class used internally to count the number of distinct values in a
    database of hierarchical objects.

    Attributes:
        - makeAccumulator: a function that takes no arguments and produces the
          Accumulator object
        - path: the path leading to this node
        - counts: the histogram OR the dictionary of sub-elements 
    """
    def __init__(self,makeAccumulator,path=None,dataset=None):
        if path is None: path=[]
        self.makeAccumulator = makeAccumulator
        self.path = path
        self.counts = None
        self.prune = False
        if dataset:
            for d in dataset:
                self.add(d)
    def pathStr(self):
        return ".".join(str(v) for v in self.path)
    def __getitem__(self,item):
        return self.counts[item]
    def setClassAttrs(self,attributes):
        """If you want to count the number of times a class' attributes
        are observed, call this to set the counted attributes before
        adding items to the counter."""
        self.counts = dict((k,HierarchicalAccumulator(self.path+[k])) for k in attributes)
    def add(self,item,callback=None):
        """Adds a new object 'item' to the hierarchical counts."""
        if self.prune: return
        if self.counts is None:
            #initialize
            if not hasattr(item,'__iter__') or isinstance(item,str):
                self.counts = self.makeAccumulator()
            elif isinstance(item,(list,tuple)):
                self.counts = [HierarchicalAccumulator(self.makeAccumulator,self.path+[i]) for i in range(len(item))]
            elif isinstance(item,dict):
                self.counts = dict((k,HierarchicalAccumulator(self.makeAccumulator,self.path+[k])) for k in item.iterkeys())
            else:
                raise TypeError("Don't know how to initialize item")
        if not hasattr(item,'__iter__') or isinstance(item,str):
            if isinstance(self.counts,accumulator.Accumulator):
                self.counts.add(item)
                if callback:
                    callback(self,item)
            else:
                assert isinstance(self.counts,dict),"Item does not follow established structure of "+self.pathStr()
                #counts is a dictionary set up to count the attributes of
                #item
                for k,c in self.counts.iteritems():
                    try:
                        i = getattr(item,k)
                    except AttributeError:
                        raise ValueError("Object does not contain attribute "+k+" in path "+self.pathStr())
                    c.add(i,callback)
        elif isinstance(item,(list,tuple)):
            assert isinstance(self.counts,(list,tuple)),"Item does not follow established structure of "+self.pathStr()
            if len(item) != len(self.counts):
                raise ValueError("Incorrect size of list in path "+self.pathStr())
            for i,c in zip(item,self.counts):
                c.add(i,callback)
        elif isinstance(item,dict):
            assert isinstance(self.counts,dict),"Item does not follow established structure of "+self.pathStr()
            if len(item) != len(self.counts):
                raise ValueError("Incorrect size of dict in path "+self.pathStr())
            for k,c in self.counts.iteritems():
                try:
                    i = item[k]
                except KeyError:
                    raise ValueError("Dict does not contain key "+k+" in path "+self.pathStr())
                c.add(i,callback)
        else:
            raise ValueError("Invalid item sent to HierarchicalValueCounter")

def AutoStats(maxDistinctValues=10):
    """A generator of pairs (datatype,hist) where datatype is either
    'mixed', 'integer', 'real', or 'string' denoting the type of input,
    and hist is a histogram of encountered values, or None if the input is mixed.

    Once maxDistinctValues is hit, the histogram is no longer updated. (This
    saves memory for large streams)
    """
    datatype = None
    h = accumulator.Histogram(maxBuckets=maxDistinctValues)
    while True:
        x = yield
        if type(x) != datatype:
            if datatype is None:
                if x is None:
                    datatype = 'mixed'
                else:
                    datatype = type(x)
            elif datatype == 'mixed':
                pass
            else:
                if isinstance(x,(str,unicode)):
                    if datatype not in (str,unicode):
                        datatype = 'mixed' 
                elif isinstance(x,(bool,int)):
                    if datatype in (str,unicode):
                        datatype = 'mixed'
                    else:
                        #promote bools to integer
                        datatype = int
                elif isinstance(x,float):
                    if datatype in (str,unicode):
                        datatype = 'mixed'
                    else:
                        #promote ints to real
                        datatype = float
                else:
                    raise ValueError("Invalid type of entry "+str(x))
        if datatype == 'mixed':
            yield (datatype,None)
        elif datatype in (str,unicode):
            h.next()
            hv = h.send(x)
            yield ('string',hv)
        elif datatype == int:
            h.next()
            hv = h.send(x)
            yield ('integer',hv)
        else:
            x = float(x)
            h.next()
            hv = h.send(x)
            yield ('real',hv)

class IncrementalFeatureMiner:
    def __init__(self,countThreshold=2,dataset=None):
        self.counter = HierarchicalAccumulator(lambda :accumulator.Accumulator(AutoStats()))
        self.countThreshold = countThreshold
        self.features = []
        self.featureTypes = []
        self.itemCount = 0
        self.newFeatureCount = 0
        if dataset is not None:
            for d in dataset:
                self.add(d)
    def add(self,item):
        self.counter.add(item,callback=self.onAddFeature)
        self.itemCount += 1
    def onAddFeature(self,counternode,item):
        hist = counternode.counts.value[1]
        if hist is None: return
        if len(hist) == self.countThreshold and not counternode.prune:
            self.features.append(counternode.path)
            self.featureTypes.append(counternode.counts.value[0])
            self.newFeatureCount += 1
            counternode.prune = True
    def getFeatureList(self):
        return self.features
    def getFeatureTypes(self):
        return self.featureTypes
    def toFeatures(self,structure):
        """For a given structure, returns the learned feature vector."""
        return extract(structure,self.features)

def _featureMine(dataset,countThreshold,quick,path):
    if len(dataset)==0: return
    item0 = dataset[0]
    if not hasattr(item0,'__iter__'):
        #base case: take a histogram
        hist = defaultdict(int)
        for i,instance in enumerate(dataset):
            try:
                hist[instance]+=1
            except KeyError:
                pathstr = '.'.join(str(v) for v in path)
                raise ValueError("Erroneous value in path "+pathstr+" entry "+str(i))
            if quick and len(hist) >= countThreshold:
                #early termination
                yield ([],len(hist))
                return 
        if len(hist) >= countThreshold:
            yield ([],hist)
        return
    elif isinstance(item0,(list,tuple)):
        #loop through entries
        n = len(item0)
        for instance in dataset:
            if len(instance) != n:
                pathstr = '.'.join(str(v) for v in path)
                raise ValueError("Not all instances of path "+pathstr+" have "+str(n)+" entries")
        for i,entries in enumerate(zip(*dataset)):
            path.append(i)
            featureIter = _featureMine(entries,countThreshold,quick,path)
            path.pop(-1)
            for p,c in featureIter:
                yield ([i]+p,c)
        return
    elif isinstance(item0,dict):
        #loop through entries
        n = len(item0)
        for instance in dataset:
            if len(instance) != n:
                raise ValueError("Not all instances of path "+path+" have "+str(n)+" entries")
        rootpaths = []
        rootcounts = []
        for key in item0.iterkeys():
            entries = []
            for i,d in enumerate(dataset):
                if key not in d:
                    raise ValueError("Item "+str(i)+" in dataset is missing key "+str(key))
                entries.append(d[key])
            path.append(key)
            featureIter = _featureMine(entries,countThreshold,quick,path)
            path.pop(-1)
            for p,c in featureIter:
                yield ([key]+p,c)
        return
    else:
        raise ValueError("Invalid structure type in dataset")

def featureMine(dataset,countThreshold=2,quick=False):
    """Given a dataset of hierarchical objects of the same structure, returns a
    those features that exhibit more than countThreshold different values
    in the dataset.

    Arguments:
    - dataset: a list of hierarchically nested values (dict, lists, or primitives)
    - countThreshold: the minimum number of distinct values needed for
      a sub-object to be counted as a feature (default 2)
    - quick: if true, does early termination when the number of distinct values
      meets countThreshold (in this case, the counts return value is not an
      accurate count).
    
    Return value: a pair (feature_paths,counts) with:
    - feature_paths: a list of feature paths, each of which can be used
      by extract/inject.
    - counts: a corresponding list of dictionaries containing histograms of
      the many distinct values observed for that feature path in the dataset.

    If the objects do not obey a fixed structure, or their values are not
    of a basic hashable datatype (bool,int,float,str) then a ValueError is
    raised.
    """
    featureiter = _featureMine(dataset,countThreshold,quick,[])
    features = [f for f in featureiter]
    if len(features)==0:
        return [],[]
    return zip(*features)

class IncrementalMultiStructureFeatureMiner:
    def __init__(self,countThreshold=2,dataset=None):
        self.structureToTemplate = dict()
        self.structureToFeatureMiner = dict()
        self.countThreshold = countThreshold
        if dataset is not None:
            for d in dataset:
                self.add(d)
    def add(self,item):
        """Adds a new item into this feature miner.  Returns the
        IncrementalFeatureMiner associated with the object"""
        s = structure(item)
        try:
            val = self.structureToFeatureMiner[s]
        except KeyError:
            #print "Adding new structure for",item,", structure",s
            val = IncrementalFeatureMiner(countThreshold=2)
            self.structureToFeatureMiner[s] = val
            self.structureToTemplate[s] = item
        val.add(item)
        return val
    def getTemplates(self):
        """Returns the list of templates"""
        return self.structureToTemplate.values()
    def getFeatureList(self,object):
        """Returns the feature paths for the given structured object.

        If the structure is not recognized, a ValueError is raised.
        """
        s = structure(object)
        try:
            f = self.structureToFeatureMiner[s].features
        except KeyError:
            raise ValueError("Structure never encountered before")
        return f
    def toFeatures(self,object):
        """Returns a pair (s,f) where s is an object structure description
        and f is a flattened feature vector.

        If the structure is not recognized, a ValueError is raised.
        """
        s = structure(object)
        try:
            f = self.structureToFeatureMiner[s].features
        except KeyError:
            raise ValueError("Structure never encountered before")
        return s,extract(object,f)
    def toObject(self,structure,features):
        """Given an object's structure, and a feature vector, returns an object
        with those features filled in.  Essentially the inverse of toFeatures.

        If the structure is not recognized, a ValueError is raised.
        """
        try:
            template = self.structureToTemplate[structure]
        except KeyError:
            raise ValueError("Structure never encountered before")
        obj = copy.deepcopy(template)
        inject(obj,template,features)
        return obj

def multiStructureFeatureMine(dataset,countThreshold=2,quick=False):
    """Given a dataset of hierarchical objects of the different structure,
    computes a list of structure templates and feature lists that exhibit
    more than countThreshold different values in the dataset.

    Arguments:
    - dataset: a list of hierarchically nested values (dict, lists, or primitives)
    - countThreshold: the minimum number of distinct values needed for
      a sub-object to be counted as a feature (default 2)
    - quick: if true, does early termination when the number of distinct values
      meets countThreshold (in this case, the counts return value is not an
      accurate count).
    
    Return value: a list of triples (template,feature_paths,counts) with:
    - template: a structure that contains the given feature paths
    - feature_paths: a list of feature paths, each of which can be used
      by extract/inject objects for structures that match that of template.
    - counts: a corresponding list of dictionaries containing histograms of
      the many distinct values observed for that feature path in the dataset.

    If the objects do not have values are not of a basic hashable datatype
    (bool,int,float,str) then a ValueError is raised.
    """
    structmatches = dict()
    for i,d in enumerate(dataset):
        s = structure(d,hashable=True)
        try:
            structmatches[s].append(d)
        except KeyError:
            structmatches[s] = [d]
    res = []
    for s,d in structmatches.iteritems():
        dres = featureMine(d,countThreshold,quick)
        f,c = dres
        res.append((d[0],f,c))
    return res


if __name__=='__main__':
    object = {'name':'Joe','type':'standard','account':1234,'orders':[2345,3456]}
    print "Account,orders:",extract(object,['account','orders'])
    print "Account,orders[1]:",extract(object,['account',['orders',1]])

    inject(object,['account','orders'],[1235,2346,3457])
    print "Injected:",object

    #test the single-structure miner
    import copy
    import time
    object1 = copy.deepcopy(object)
    object2 = copy.deepcopy(object1)
    object2['name'] = 'Mary'
    object2['account'] = 586
    object2['orders'][1] = 7201
    t0 = time.time()
    print "Features:",featureMine([object1,object2]*10000,quick=True)[0]
    print "Batch time:",time.time()-t0
    t0 = time.time()
    disc = IncrementalFeatureMiner(dataset=[object1,object2]*10000)
    t1 = time.time()
    print "Features:",disc.getFeatureList()
    print "Incremental time:",t1-t0

    #test the multi-structure miner
    t0 = time.time()
    structs,featureLists,featureCounts = zip(*multiStructureFeatureMine([100,40,{'foo':'bar'},{'foo':'baz','a':30},{'foo':'baz','a':60},40]*10000))
    t1 = time.time()
    print "Template: Features"
    for (s,f) in zip(structs,featureLists):
        print structure(s,hashable=False),":",",".join(str(fi) for fi in f)
    print "Batch time",t1-t0
    t0 = time.time()
    disc = IncrementalMultiStructureFeatureMiner(dataset=[100,40,{'foo':'bar'},{'foo':'baz','a':30},{'foo':'baz','a':60},40]*10000)
    t1 = time.time()
    templates = disc.getTemplates()
    print "Template: Features"
    for t in templates:
        f = disc.getFeatureList(t)
        print structure(t,hashable=False),':',",".join(str(fi) for fi in f)
    print "Incremental time",t1-t0    
                                                                       
