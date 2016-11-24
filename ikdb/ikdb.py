import pkg_resources
if pkg_resources.get_distribution('klampt').version >= '0.7':
    from klampt.model import ik
    from klampt.math import vectorops
else:
    from klampt import ik,vectorops
import random
import numpy as np
import time
import copy
import features
import metriclearning
from metriclearning import mahalanobis_distance2
import functionfactory
from utils import *
from ikproblem import *
from dynamicarray import *
import os
import math



class IKDatabase:
    """Attributes:
    - robot: the robot used
    - ikTemplate: an Json object instance used as a template for all problems
      in this database.
    - featureNames: a list of features of the IK problem
    - featureRanges: a list of feature ranges (a,b) that are valid for the
      IK problems of this database.
    - problems: a list of IKProblems in the database.  Usually empty to save
      memory.  Use numProblems() and getProblem(i) in lieu of len(problems) and
      problems[i].
    - solutions: a list of problem solutions in the database
    - numSolutionsToTry: the default number of solutions to try during solve()
    - problemFeatures: a list of problem features
    - numQueries: the total number of solve() queries
    - lookupTime: the total time taken to find the closest k problems
    - solveTime: the total time used in the IK solvers
    - metricMatrix: a matrix A for the mahalanobis distance, or None for Euclidean
    - metricTransform: a matrix L such that LL^T = A for the mahalanobis distance, or None for Euclidean

    To learn a database offline, follow the following steps:
    1. create a template IK problem:
       template = IKProblem()
       #TODO: setup template by adding IK objectives, cost functions, etc
    2. Now create a set of features, such as...
       featurename1 = ('objectives',0,'endPosition')
    3. And set up a set of ranges, such as
       featureranges = [(-1,1),(-1,1),(-1,1)]
    4. Finally, create the DB with random problems and the random-restart global solver
       db = IKDatabase(robot)
       db.setIKProblemSpace(template,[featurename1,featurename2,...],featureranges)
       randomproblems = [db.sampleRandomProblem() for i in range(numProblems)]
       db.generate(randomproblems)

    You may now save the database and its metadata to disk.  If your feature space is badly scaled,
    you will want to run metric learning as follows:
       db.metricLearn(N)
    where N is something like db.numProblems()*50.

    To save/load the database property you will need to call both save/load and
    saveMetadata/loadMetadata.
    
    The database format on disk is a simple file with alternating lines:
    [problem features 1]
    [solution to problem 1]
    [problem features 2]
    [solution to problem 2]
    ...
    each of the lines is a whitespace-separated list of entries.

    On load(), problems, ikTemplate, featureNames, and featureRanges are not
    populated propery.  You will need to call loadMetadata to get the problems
    back.

    For online use, you will need to set up the database with these steps:
       db = IKDatabase(robot)
       db.load(db_fn)  #previously saved database file
       db.loadMetadata(dbmetadata_fn)   #previously saved metadata file
       db.numSolutionsToTry = 10  #this is the value of k in the algorithm

    Now for each new problem, call
       (soln,prim) = db.solve(newProblem)
    """
    def __init__(self,robot,problems=None,solutions=None):
        self.robot = robot

        self.ikTemplate = None
        self.featureNames = None
        self.featureRanges = None

        if problems is None: problems = []
        if solutions is None: solutions = []
        self.problems = problems
        self.solutions = solutions
        self.numSolutionsToTry = 1

        self.problemFeatures = DynamicArray2D()
        self.feasiblePredictionLearning = None
        self.metricMatrix = None
        self.metricTransform = None
        self.numQueries = 0
        self.lookupTime = 0
        self.solveTime = 0
        self.nn = None
        self.nnBuildSize = 0

        import threading
        self.lock = threading.RLock()
        return

    def numProblems(self):
        """Returns the number of problems in the database"""
        return len(self.solutions)

    def resetStats(self):
        self.lookupTime = self.solveTime = 0
        self.numQueries = 0
        
    def stats(self):
        return {'numQueries':self.numQueries,'lookupTime':self.lookupTime,'solveTime':self.solveTime}        

    def save(self,fn):
        """Saves the database from the given file."""
        with self.lock:
            print "Saving IKDatabase data to",fn
            assert(len(self.problemFeatures) == len(self.solutions))
            with open(fn,'w') as f:
                for i,s in enumerate(self.solutions):
                    f.write(' '.join(str(v) for v in self.problemFeatures[i])+'\n')
                    if s is None:
                        f.write('\n')
                    else:
                        f.write(' '.join(str(v) for v in s)+'\n')
        return

    def load(self,fn):
        """Loads the database from the given file."""
        with self.lock:
            with open(fn,'r') as f:
                self.problemFeatures = DynamicArray2D()
                self.solutions = []
                while True:
                    fline = f.readline()
                    if not fline: break
                    sline = f.readline()
                    if not sline:
                        raise IOError("Database file has odd number of lines?")
                    fline = fline.strip()
                    sline = sline.strip()
                    features = [float(v) for v in fline.split()]
                    self.problemFeatures.append(features)
                    if len(sline)==0:
                        self.solutions.append(None)
                    else:
                        self.solutions.append([float(v) for v in sline.split()])
            self.problemFeatures.compress()
        print "IKDatabase: Read",len(self.solutions),"solutions from file",fn
        assert len(self.solutions)==len(self.problemFeatures)
        t1 = time.time()
        self.buildNNDataStructure()
        t2 = time.time()
        print "IKDatabase: Built query data structure in %fs"%(t2-t1,)

    def saveMetadata(self,fn):
        """Saves metadata about this database to fn, in JSON format"""
        import json
        print "Saving IKDatabase metadata to",fn
        metadata = {}
        metadata['type'] = 'IKDatabase'
        metadata['robot'] = self.robot.getName()
        metadata['stats'] = self.stats()
        if self.ikTemplate is not None:
            metadata['ikTemplate'] = self.ikTemplate
        if self.featureNames is not None:
            metadata['featureNames'] = self.featureNames
        if self.featureRanges is not None:
            metadata['featureRanges'] = self.featureRanges
        else:
            print "Saving metadata... No feature ranges?"
        if self.feasiblePredictionLearning:
            metadata['feasiblePredictionLearning'] = self.feasiblePredictionLearning
        if self.metricMatrix is not None:
            metadata['metricMatrix'] = self.metricMatrix.tolist()
        with open(fn,'w') as f:
            json.dump(metadata,f)
        return

    def loadMetadata(self,fn):
        """Loads metadata about this database to fn, in JSON format.  Raises
        IOError on exception"""
        import json
        with open(fn,'r') as f:
            obj = json_byteify(json.load(f))
        if obj.get('type') != 'IKDatabase':
            raise IOError("Metadata is not of IKDatabase type")
        if obj.get('robot') != self.robot.getName():
            raise IOError("Metadata does not match robot name")
        with self.lock:
            if 'ikTemplate' in obj:
                self.ikTemplate = obj['ikTemplate']
            if 'featureNames' in obj:
                self.featureNames = obj['featureNames']
            if 'featureRanges' in obj:
                self.featureRanges = obj['featureRanges']
            if 'feasiblePredictionLearning' in obj:
                self.feasiblePredictionLearning = obj['feasiblePredictionLearning']
            if self.ikTemplate is not None and self.featureNames is not None:
                f = self.featureNames
                self.featureNames = None
                self.setIKProblemSpace(self.ikTemplate,f,self.featureRanges)
            if 'metricMatrix' in obj:
                self.metricMatrix = np.array(obj['metricMatrix'])
                self.metricTransform = np.linalg.cholesky(self.metricMatrix)
        return

    def compress(self):
        """Takes up less memory by erasing problems. Useful when you want to
        freeze a database and just make queries.  Certain calls, like generate,
        will automatically decompress the database.
        """
        with self.lock:
            del self.problems
            self.problems = []
            self.problemFeatures.compress()

    def decompress(self):
        """Takes up more memory"""
        with self.lock:
            self.problemFeatures.decompress()

    def setIKProblemSpace(self,template,featureNames,featureRanges=None):
        """Sets the problem space for this database."""
        with self.lock:
            oldIkTemplate = self.ikTemplate
            oldFeatureNames = self.featureNames
            #this line does some debugging...
            f = ikProblemToFeatures(template,featureNames)
            if featureRanges is not None:
                assert(len(f) == len(featureRanges)),"Feature space has dimension %d while %d ranges were provided"%(len(f),len(featureRanges))
            if isinstance(template,IKProblem):
                self.ikTemplate = template.toJson()
            else:
                self.ikTemplate = template
            self.featureNames = featureNames
            self.featureRanges = featureRanges

            t0 = time.time()
            if len(self.problems) > 0:
                #rebuild problem feature list and nearest neighbor structure
                print "Building problem feature list, first time..."
                self.problemFeatures = DynamicArray2D([self.problemToFeatures(p) for p in self.problems])
                self.problems = []
            elif len(self.problemFeatures) > 0:
                if oldFeatureNames is not None:
                    #rebuild problems from old features
                    print "Rebuilding problem feature list..."
                    n = len(self.problemFeatures)
                    self.problemFeatures.decompress()
                    assert len(self.problemFeatures) == n
                    assert oldIkTemplate is not None,"Can't rebuild features because there wasn't a previous IK template?"
                    cnt = 0
                    for i,f in enumerate(self.problemFeatures):
                        assert len(oldFeatureNames) == len(f)
                        p = featuresToIkProblem(oldIkTemplate,oldFeatureNames,f)
                        self.problemFeatures[i] = self.problemToFeatures(p)
                        assert len(self.problemFeatures[i]) == len(self.featureNames)
                        cnt += 1
                    assert cnt == len(self.problemFeatures)
                    assert cnt == n
                else:
                    assert len(self.problemFeatures[0]) == len(self.featureNames),"Pre-existing problem feature array isn't the right size?"
        self.buildNNDataStructure()
        t1 = time.time()
        print "IKDatabase: Rebuilt features and NN data structure in %fs"%(t1-t0,)
        #sanity check
        for p in self.problemFeatures:
            assert len(p) == len(self.featureNames)

    def autoSetFeatureRanges(self):
        """Sets the featureRanges variable from the ranges of feasible
        solutions"""
        with self.lock:
            if self.ikTemplate.get('softObjectives',False):
                fmin,fmax = None,None
                for p,s in zip(self.problemFeatures,self.solutions):
                    if s is None: continue
                    if fmin == None or any(v < a or v > b for (v,a,b) in zip(p,fmin,fmax)):
                        problem = featuresToIkProblem(self.ikTemplate,self.featureNames,p)
                        self.robot.setConfig(s)
                        resid = problem.constraintResidual(self.robot)
                        #TODO: get the actual tolerance of successful solves?
                        tol = 1e-3
                        if all(abs(v) < tol for v in resid):
                            if fmin == None:
                                fmin = p[:]
                                fmax = p[:]
                            else:
                                for i,(v,a,b) in enumerate(zip(p,fmin,fmax)):
                                    if v < a:
                                        fmin[i] = v
                                    elif v > b:
                                        fmax[i] = v
                print "Feature ranges set to",fmin,fmax
                self.featureRanges = zip(fmin,fmax)
            else:
                feasfeatures = [p for p,s in zip(self.problemFeatures,self.solutions) if s is not None]
                if len(feasfeatures)==0:
                    print "IKDatabase.autoSetFeatureRanges(): no feasible solutions"
                    #can't set
                    return
                feasfeatures = np.array(feasfeatures)
                fmax = np.amax(feasfeatures,axis=0).tolist()
                fmin = np.amin(feasfeatures,axis=0).tolist()
                self.featureRanges = zip(fmin,fmax)

    def sampleRandomProblem(self,featureExpandAmountRel=0,featureExpandAmountAbs=0):
        """Samples a random problem.  The default implementation uses
        the IK templates and ranges provided to setIKProblemSpace, and samples
        uniformly from those ranges.

        featureExpandAmountRel/Abs optionally allows you to expand the feature
        ranges from the provided range.  The sampled interval [a',b'] is
        a' = a - 0.5*(featureExpandAmountRel*(b-a)+featureExpandAmountAbs)
        b' = b + 0.5*(featureExpandAmountRel*(b-a)+featureExpandAmountAbs)
        where [a,b] are the provided interval
        """
        if self.featureRanges is None:
            raise RuntimeError("IKDatabase: did not set up feature ranges")
        if self.ikTemplate is None:
            raise RuntimeError("IKDatabase: did not set up IK template")
        x = [random.uniform(a-0.5*(featureExpandAmountRel*(b-a)+featureExpandAmountAbs),b+0.5*(featureExpandAmountRel*(b-a)+featureExpandAmountAbs)) for (a,b) in self.featureRanges]
        return featuresToIkProblem(self.ikTemplate,self.featureNames,x)

    def getProblem(self,index):
        """Retrieves a previous IKProblem.  You should call this instead of
        self.problems[index].
        """
        with self.lock:
            if len(self.problems) > 0: return self.problems[index]
            elif len(self.problemFeatures) > 0:
                if index < 0 or index >= len(self.problemFeatures):
                    print "Going to abort... data",index,len(self.problemFeatures)
                assert index >= 0 and index < len(self.problemFeatures)
                return featuresToIkProblem(self.ikTemplate,self.featureNames,self.problemFeatures[index])
            else:
                print len(self.problems),len(self.problemFeatures),index
                raise ValueError("Invalid problem index")
        
    def problemToFeatures(self,problem):
        """Can overload this to implement custom feature maps. By default calls
        the ikProblemToFeatures function."""
        with self.lock:
            res = ikProblemToFeatures(problem,self.featureNames)
        return res

    def generate(self,problemList,solverParams=None):
        """Generates additional instances to the problem database.
        - problemList: a list of IK problem instances
        - solverParams: an IKSolverParams structure passed to self.solveRaw(),
          or none for default parameters (see self.solveRaw() for more details)
        """
        if solverParams is None: solverParams = IKSolverParams()
        t0 = time.time()
        solutionList = [self.solveRaw(p,solverParams) for p in problemList]
        t1 = time.time()
        numSolved = len([s for s in solutionList if s is not None])
        print "IKDatabase: Solved %d/%d problems in %fs"%(numSolved,len(problemList),t1-t0)
        with self.lock:
            self.solutions += solutionList
            if self.ikTemplate is None:
                self.problems += problemList
            else:
                self.problemFeatures += [self.problemToFeatures(p) for p in problemList]
        self.buildNNDataStructure()
        t2 = time.time()
        print "IKDatabase: Built query data structure in %fs"%(t2-t1,)

    def balance(self,fracPositive=0.5):
        """Balances the database so that approximately fracPositive fraction
        of the examples are positive (have a solution)"""
        if self.ikTemplate is None:
            posExamples = [(p,s) for p,s in zip(self.problems,self.solutions) if s is not None]
            negExamples = [(p,s) for p,s in zip(self.problems,self.solutions) if s is None]
        else:
            posExamples = [(p,s) for p,s in zip(self.problemFeatures,self.solutions) if s is not None]
            negExamples = [(p,s) for p,s in zip(self.problemFeatures,self.solutions) if s is None]
        if len(posExamples) >= fracPositive*self.numProblems():
            #subsample positive examples
            n = min(int(fracPositive*self.numProblems()),len(posExamples))
            print "Subsampling",n,"/",len(posExamples),"positive examples"
            posExamples = random.sample(posExamples,n)
        else:
            #subsample negative examples
            n = min(int((1.0-fracPositive)*self.numProblems()),len(negExamples))
            print "Subsampling",n,"/",len(negExamples),"negative  examples"
            negExamples = random.sample(negExamples,n)
        self.solutions = [e[1] for e in posExamples]+[e[1] for e in negExamples]
        if self.ikTemplate is None:
            self.problems = [e[0] for e in posExamples]+[e[0] for e in negExamples]
        else:
            self.problemFeatures = DynamicArray2D([e[0] for e in posExamples]+[e[0] for e in negExamples])
        t1 = time.time()
        self.buildNNDataStructure()
        t2 = time.time()
        print "IKDatabase: Built query data structure in %fs"%(t2-t1,)
        
    def solve(self,problem,numSolutionsToTry=None,params=None,features=None,tryCurrent=False):
        """Solves an instance of a problem using the database.
        - problem: the IKProblem instance
        - numSolutionsToTry: if provided, the number of solutions to try and
          adapt to the new problem.  By default, uses self.numSolutionsToTry.
        - params: if provided, the IKSolverParams to use.  By default, uses
          the default IKSolverParams constructor.
        - features: if provided, the feature array provided by
          np.array(self.problemToFeatures(problem)).  Avoids some minor overhead.
        - tryCurrent: if true, tries the current configuration if it has a
          better score than the best example.
        Return value (soln,adapted or best primitive):
        - soln: the solution configuration or None if failed
        - adapted or best primitive: the index of the adapted primitive, or the
          closest non-adapted primitive.
        """
        if params is None:
            #the defaults are good here
            params = IKSolverParams()
        if numSolutionsToTry is None: numSolutionsToTry = self.numSolutionsToTry
        if features is None:
            features = np.array(self.problemToFeatures(problem))
        self.numQueries += 1
        t0 = time.time()
        dist,ind = self.knn(features,numSolutionsToTry)
        t1 = time.time()
        self.lookupTime += t1-t0

        if tryCurrent:
            q0 = self.robot.getConfig()
            quality0 = problem.score(self.robot)
        closestTried = -1
        for index,i in enumerate(ind):
            if self.solutions[i] is not None:
                self.robot.setConfig(self.solutions[i])
                if tryCurrent:
                    if problem.score(self.robot) > quality0:
                        #print "current has better quality",quality0,problem.score(self.robot)
                        continue
                    #print "example has better quality",problem.score(self.robot),quality0
                res = self.solveAdapt(problem,params)
                if res is not None:
                    t2 = time.time()
                    self.solveTime += t2-t1
                    return (res,index)
                if closestTried == -1: closestTried = i
        if tryCurrent:
            self.robot.setConfig(q0)
            res = self.solveAdapt(problem,params)
            t2 = time.time()
            self.solveTime += t2-t1
            return (res,-1)
        t2 = time.time()
        self.solveTime += t2-t1
        return None,closestTried

    def predictFeasible(self,problem,numSolutionsToQuery,recallBias=0.5,features=None):
        """Predicts whether a problem is feasible.  Returns a pair
        (prediction,confidence) where confidence is in the range [0,1].
        - problem:  the IKProblem instance
        - numSolutionsToQuery: the number of solutions to query during the
          prediction.
        - recallBias: can bias the prediction to have higher recall / higher
          precision by choosing a larger / smaller value.  In range [0,1].
        - features: if provided, the feature array provided by
          np.array(self.problemToFeatures(problem)).  Avoids some minor overhead.
        """
        if features is None:
            features = np.array(self.problemToFeatures(problem))
        if self.feasiblePredictionLearning is None or self.feasiblePredictionLearning['k'] != numSolutionsToQuery or recallBias != self.feasiblePredictionLearning['recallBias']:
            self.runFeasiblePredictionLearning(numSolutionsToQuery,recallBias)
        t0 = time.time()
        dist,ind = self.knn(features,numSolutionsToQuery)
        t1 = time.time()
        self.lookupTime += t1-t0
        numFeasible = 0
        for i in ind:
            if self.solutions[i] is not None:
                numFeasible += 1
        threshold = self.feasiblePredictionLearning['threshold']
        accuracyMap = self.feasiblePredictionLearning['confidence']
        #print "distances:",dist
        #print "fraction feasible:",float(numFeasible)/len(ind),"confidence",accuracyMap[numFeasible]
        return numFeasible > len(ind)*threshold, accuracyMap[numFeasible]

    def clearFeasiblePredictionLearning(self):
        """If you have added several problems to the database, you may want to call this to refresh
        the predictFeasible calls"""
        self.feasiblePredictionLearning = None      

    def runFeasiblePredictionLearning(self,numSolutionsToQuery,recallBias):
        """This is called the first time you call predictFeasible, to predict the probability of
        infeasibility for new problems.  Performs LOO cross-validation.  You may call this to
        pre-cache the LOO results."""
        with self.lock:
            self.feasiblePredictionLearning = {}
            if len(self.problemFeatures) <= numSolutionsToQuery:
                self.feasiblePredictionLearning['N'] = len(self.problemFeatures)
                self.feasiblePredictionLearning['k'] = numSolutionsToQuery
                self.feasiblePredictionLearning['recallBias'] = recallBias
                self.feasiblePredictionLearning['threshold'] = recallBias
                self.feasiblePredictionLearning['confidence'] = [0.5]*(numSolutionsToQuery+1)
                return
            if self.nn is not None or self.nnBuildSize < numSolutionsToQuery:
                #need to rebuild nn data structure
                self.buildNNDataStructure()
                if self.nn is None:
                    print len(self.problemFeatures)
                    print self.featureNames
                    print self.ikTemplate
                    raise ValueError("Can't do feasibility prediction learning, no NNdata structure available")
            self.feasiblePredictionLearning['N'] = len(self.problemFeatures)
            self.feasiblePredictionLearning['k'] = numSolutionsToQuery
            self.feasiblePredictionLearning['recallBias'] = recallBias
            print "Running LOO cross validation for %d-NN classification, bias %f..."%(numSolutionsToQuery,recallBias)
            #run LOO cross validation
            try:
                farray = self.problemFeatures.array
            except AttributeError:
                farray = np.array(self.problemFeatures.items)
            dist,ind = self.nn.query(farray,numSolutionsToQuery+1)
            labelList = {}
            for i in range(numSolutionsToQuery+1):
                labelList[float(i)/numSolutionsToQuery] = [1,1]
            for i in xrange(len(self.problemFeatures)):
                inds = ind[i,1:numSolutionsToQuery+1]
                labels = [(self.solutions[j] is not None) for j in inds]
                frac = float(sum(1 if i else 0 for i in labels))/numSolutionsToQuery
                if self.solutions[i] is not None: #feasible
                    labelList[frac][1] += 1
                else: #infeasible
                    labelList[frac][0] += 1
            keys = sorted(labelList.keys())
            #find the threshold that obtains the minimum recall-weighted error
            #E = FN*(recallBias) + FP*(1-recallBias)
            threshold = 0
            count = 0
            fp = 0
            fn = 0
            #count false negatives
            for k,v in labelList.iteritems():
                fp += v[0]
                count += v[0]+v[1]
            i = 0
            #print "Initial FP",fp,"FN",fn
            bestThreshold = 0
            bestLoss = fn*recallBias + fp*(1-recallBias)
            bestAccuracy = count-(fn+fp)
            threshold = (keys[0]+keys[1])*0.5
            while i < len(keys):
                while i < len(keys) and keys[i] <= threshold:
                    fn += labelList[keys[i]][1]
                    fp -= labelList[keys[i]][0]
                    i+=1
                loss = fn*recallBias + fp*(1-recallBias)
                #print "Threshold",threshold,"FP",fp,"FN",fn,"loss",loss
                if loss < bestLoss:
                    bestThreshold = threshold
                    bestLoss = loss
                    bestAccuracy = count-(fn+fp)
                if i == len(labelList):
                    #done, reached end of lists
                    break
                else:
                    #not done, get the next threshold
                    if i+1 < len(keys):
                        threshold = (keys[i]+keys[i+1])*0.5
                    else:
                        threshold = 1
            self.feasiblePredictionLearning['threshold'] = bestThreshold
            accuracyMap = [0]*(numSolutionsToQuery+1)
            for k in keys:
                neg,pos = labelList[k]
                index = int(numSolutionsToQuery*k)
                if k < bestThreshold:
                    accuracyMap[index] = float(neg)/float(neg+pos)
                else:
                    accuracyMap[index] = float(pos)/float(neg+pos)
            self.feasiblePredictionLearning['confidence'] = accuracyMap
        print "  Resulting theshold",bestThreshold,"accuracy",float(bestAccuracy)/count
        print "  Accuracy map:",accuracyMap

    def solveRaw(self,problem,params):
        """Solves an instance of a problem from scratch, or returns None
        for no solution.
        
        Arguments:
            - problem: an IKProblem, an IKObjective, or list of IKObjectives.
            - params: an IKSolverParams instance
        Result:
            - the solution configuration, or None if the problem was not solved.
        """
        if isinstance(problem,IKProblem):
            return problem.solve(self.robot,params)
        else:
            return IKProblem(problem).solve(self.robot,params)

    def solveAdapt(self,problem,params):
        """Given the robot already in a solution to a prior problem,
        tries to solve the given problem.  Returns the robot's configuration, or
        None if the problem was not solved.
        """
        assert params.startRandom == False, "IKSolverParams.startRandom should be false for an adaptation solver"
        assert params.numRestarts == 1, "IKSolverParams.numRestarts should be 1 for an adaptation solver"
        if isinstance(problem,IKProblem):
            return problem.solve(self.robot,params)
        else:
            print "Warning: creating IKProblem from raw IKObjectives?"
            return IKProblem(problem).solve(self.robot,params)

    def buildNNDataStructure(self):
        """Builds a nearest neighbor data structure.  User doesn't need to
        call this unless the self.problems attribute was changed manually."""
        if len(self.problemFeatures)==0 or len(self.featureNames)==0:
            return
        try:
            from sklearn.neighbors import NearestNeighbors,BallTree
            from scipy.spatial import KDTree
            with self.lock:
                try:
                    farray = self.problemFeatures.array
                except AttributeError:
                    farray = np.array(self.problemFeatures.items)
                if self.metricTransform is not None:
                    farray = np.dot(farray,self.metricTransform)
                #self.nn = NearestNeighbors(n_neighbors=1,algorithm="auto").fit(farray)
                self.nn = BallTree(farray)
                #self.nn = KDTree(farray)
                self.nnBuildSize = len(self.problemFeatures)
        except ImportError:
            print "IKDatabase: Warning, scikit-learn is not installed, queries will be much slower"
            with self.lock:
                self.nn = None
                self.nnBuildSize = 0
        return

    def knn(self,features,k):
        """Returns the k-nearest neighbors to the feature vector features.
        Result is a pair (dist,ind) where dist is a list of distances and
        ind is a list of problem indices, in order of increasing distance.
        """
        with self.lock:
            if self.nn:
                k = min(k,self.nnBuildSize)
                qvec = features if self.metricTransform is None else np.dot(self.metricTransform.T,features)
                #this line is needed for scipy 0.17 and above...
                dist,ind = self.nn.query(qvec.reshape(1,-1),k)
                #dist,ind = self.nn.query(qvec,k)
                #scipy KD tree
                #if k > 1:
                #    dist = dist[0,:].tolist()
                #    ind = ind[0,:].tolist()
                #ball tree
                dist = dist[0,:].tolist()
                ind = ind[0,:].tolist()
                
                if self.nnBuildSize < len(self.problemFeatures):
                    #some extras, take them into account
                    dmax,imax = max((d,i) for (i,d) in enumerate(dist))
                    for i in xrange(self.nnBuildSize,len(self.problemFeatures)):
                        d = mahalanobis_distance2(features,self.problemFeatures[i],self.metricMatrix)
                        if d < dmax**2:
                            dist[imax] = math.sqrt(d)
                            ind[imax] = i
                            dmax,imax = max((d,i) for (i,d) in enumerate(dist))
                    di = zip(dist,ind)
                    dist,ind = zip(*sorted(di))
                return dist,ind
            else:
                #no Scikit-learn installed... do brute force nearest neighbors?
                if len(self.problemFeatures)==0:
                    return [],[]
                if k > math.log(len(self.problemFeatures)):
                    #just sort, O(n log n + k) time
                    di = [(mahalanobis_distance2(features,f,self.metricMatrix),i) for i,f in enumerate(self.problemFeatures)]
                    di = sorted(di)
                    #TODO: return sqrt of distance squared?
                    return zip(*di[:min(k,len(self.problemFeatures))])
                else:
                    #maintain top k as you loop through, O(k*n) time
                    dist = [mahalanobis_distance2(features,self.problemFeatures[i],self.metricMatrix) for i in xrange(k)]
                    ind = range(k)
                    dmax,imax = max((d,i) for (i,d) in enumerate(dist))
                    for i in xrange(k,len(self.problemFeatures)):
                        d = mahalanobis_distance2(features,self.problemFeatures[i],self.metricMatrix)
                        if d < dmax:
                            dist[imax] = d
                            ind[imax] = i
                            dmax,imax = max((d,i) for (i,d) in enumerate(dist))
                    di = zip(dist,ind)
                    dist,ind = zip(*sorted(di))
                    return [math.sqrt(d) for d in dist],ind

    def revise(self):
        """Revises the database to reduce noise in generating solutions."""
        numNew = 0
        numBetter = 0
        print "IKDatabase: Revising database..."
        for i in xrange(self.numProblems()):
            p = self.getProblem(i)
            s,prim = self.solve(p,numSolutionsToTry=10)
            if s is None: continue
            if self.solutions[i] is None:
                self.solutions[i] = s
                numNew += 1
            elif p.costFunction(s) < p.costFunction(self.solutions[i]) - 1e-5:
                self.solutions[i] = s
                numBetter += 1
        print "  Found",numNew,"new solutions, improved",numBetter,"solutions"

    def metricLearnStep(self,features1,features2,solution1,solution2):
        """Performs 1 step of metric learning"""
        if self.metricMatrix is None:
            self.metricMatrix = np.eye(len(features1))
        if (solution1 is None) != (solution2 is None):
            sign = -1
        else:
            sign = 1
        rate = 0.1
        self.metricMatrix = metriclearning.metric_logdet_update(self.metricMatrix,np.array(features1),np.array(features2),1,sign,rate)
        self.metricTransform = np.linalg.cholesky(self.metricMatrix)

    def metricLearn(self,numIters):
        """Runs numIters iterations of metric learning."""
        pPickRandom = 0.5
        feasibleSet = [i for i in xrange(len(self.solutions)) if self.solutions[i] is not None]
        if len(feasibleSet) == 0:
            #no data
            return
        for iters in xrange(numIters):
            i = random.choice(feasibleSet)
            primitives = []
            s = self.solutions[i]
            p = self.getProblem(i)
            if random.random() < pPickRandom:
                #pick random
                primitives.append(random.randint(0,len(self.solutions)-1))
            else:
                #pick nearby
                dist,ind = self.knn(np.array(self.problemFeatures[i]),20)
                feas = [j for j in ind if i!=j and self.solutions[j] is not None]
                infeas = [j for j in ind if i!=j and self.solutions[j] is None]
                for j in infeas:
                    self.metricLearnStep(self.problemFeatures[j],self.problemFeatures[i],None,s)
                if len(feas) > 0:
                    primitives = feas
            for primitive in primitives:
                if self.solutions[primitive] is None:
                    #push farther away
                    self.metricLearnStep(self.problemFeatures[primitive],self.problemFeatures[i],None,s)
                else:
                    #try adapting
                    self.robot.setConfig(self.solutions[primitive])
                    res = self.solveAdapt(p,IKSolverParams())
                    if res:
                        subopt = p.costFunction(res) - p.costFunction(s)
                        if subopt > 0.1:
                            res = None
                    self.metricLearnStep(self.problemFeatures[primitive],self.problemFeatures[i],self.solutions[primitive],res)
        return

class IKTestInstance:
    def __init__(self,problem,feasible=None,cost=None):
        self.problem=problem
        self.feasible=feasible
        self.cost=cost

class IKTestSet:
    """A helper for cross-validation / holdout set validation.  Stores a bunch of "ground truth"
    optimal solutions generated by the optimized random-restart method."""
    def __init__(self,robot):
        self.groundTruthSolveParams = IKSolverParams(numIters=50,numRestarts=100,localMethod='auto')
        #Uncomment to use off-the-shelf global method. Currently much worse than custom method.
        #self.groundTruthSolveParams = IKSolverParams(numIters=6000,numRestarts=100,globalMethod='DIRECT')
        #self.groundTruthSolveParams = IKSolverParams(globalMethod='scipy.differential_evolution')
        self.robot = robot
        self.testSet = None

    def loadTest(self,fn):
        """Loads test instances with labels to a file: expanded version"""
        print "Loading test instances from",fn
        import json
        f = open(fn,'r')
        self.testSet = []
        for line in f.readlines():
            test_instance = json_byteify(json.loads(line))
            problem = test_instance['problem']
            feasible = test_instance.get('feasible',-1)
            cost = test_instance.get('cost',None)
            if feasible < 0:
                feasible = None
            else:
                feasible = bool(feasible)
            if cost is not None:
                cost = float(cost)
            p = IKProblem()
            p.fromJson(problem)
            self.testSet.append(IKTestInstance(p,feasible,cost))
        f.close()
        print "Loaded",len(self.testSet),"test instances"
        print "  ",sum(1 for i in self.testSet if i.feasible == True),"feasible, ",sum(1 for i in self.testSet if i.feasible==False),"infeasible, ",sum(1 for i in self.testSet if i.feasible is None),"unknown"
        return

    def loadTestFeatures(self,fn,ikTemplate,featureList):
        """Loads the test set as a feature list.  Saves some space on disk
        compared to saveTest."""
        print "Loading test instances from",fn
        f = open(fn,'r')
        self.testSet = []
        for line in f.readlines():
            values = [float(v) for v in line.split()]
            featureVec = values[:-2]
            problem = featuresToIkProblem(ikTemplate,featureList,featureVec)
            feasible = values[-2]
            cost = values[-1]
            if feasible < 0:
                feasible = None
            else:
                feasible = bool(feasible)
            cost = float(cost)
            self.testSet.append(IKTestInstance(problem,feasible,cost))
        f.close()
        print "Loaded",len(self.testSet),"test instances"
        print "  ",sum(1 for i in self.testSet if i.feasible == True),"feasible, ",sum(1 for i in self.testSet if i.feasible==False),"infeasible, ",sum(1 for i in self.testSet if i.feasible is None),"unknown"
        return

    def saveTestFeatures(self,fn,featureNames):
        """Saves the test set as a feature list.  Saves some space on disk
        compared to saveTest."""
        f = open(fn,'w')
        for instance in self.testSet:
            print instance.feasible
            v = ikProblemToFeatures(instance.problem,featureNames)
            v.append(int(instance.feasible) if instance.feasible is not None else -1)
            v.append(instance.cost if instance.cost is not None else 0)
            f.write(" ".join(str(x) for x in v)+'\n')
        f.close()
        return


    def saveTest(self,fn):
        """Saves test instances with labels to a file"""
        import json
        f = open(fn,'w')
        for inst in self.testSet:
            obj = {'problem':inst.problem.toJson()}
            if inst.feasible is not None:
                obj['feasible'] = int(inst.feasible)
            if inst.cost is not None:
                obj['cost'] = inst.cost
            json.dump(obj,f)
            f.write('\n')
        f.close()
        return

    def generateTestLabels(self,problemList):
        """Generates this dataset, with labels, for the given problem list
        using the current value of groundTruthSolveParams."""
        self.testSet = [IKTestInstance(p) for p in problemList]
        print "**** Generating ground truth labels ***"
        numFeasible,numInfeasible=0,0
        feasibleTime,infeasibleTime = 0,0
        costSum = 0
        costCount = 0
        for p in self.testSet:
            t0 = time.time()
            res = p.problem.solve(self.robot,self.groundTruthSolveParams)
            t1 = time.time()
            if res is not None:
                feasibleTime += t1-t0
                p.feasible = True
                if p.problem.costFunction is not None:
                    p.cost = p.problem.costFunction(res)
                    costSum += p.cost
                    costCount += 1
                numFeasible += 1
            else:
                p.feasible = False
                numInfeasible += 1
                infeasibleTime += t1-t0
        print "IK with 100 random restarts: solved %d/%d in time %f"%(numFeasible,len(self.testSet),feasibleTime+infeasibleTime)
        numFeasible = max(numFeasible,1)
        numInfeasible = max(numInfeasible,1)
        print "  Average time for feasible %f, infeasible %f"%(feasibleTime/numFeasible,infeasibleTime/numInfeasible)
        if costCount > 0:
            print "  Average cost function value",costSum / costCount

    def test(self,solveFunc,name):
        """Generic tester of a solver function.  solveFunc takes in an
        IKProblem and returns a solution.  name is the name of the solve
        method, for use in the printed output."""
        feasibleTime,infeasibleTime = 0,0
        costSuboptimality = 0
        numCostsCounted = 0
        numFeasible,numInfeasible = 0,0
        numFeasibleSolved,numInfeasibleSolved = 0,0
        for p in self.testSet:
            t0 = time.time()
            res = solveFunc(p.problem)
            t1 = time.time()
            if p.feasible == True:
                feasibleTime += t1-t0
                numFeasible += 1
            else:
                infeasibleTime += t1-t0
                numInfeasible += 1
            if res is not None:
                if p.problem.costFunction is not None and p.cost is not None:
                    costSuboptimality += p.problem.costFunction(res) - p.cost
                    numCostsCounted += 1
                if p.feasible == True:
                    numFeasibleSolved += 1
                else:
                    numInfeasibleSolved += 1
        print "IKTestSet method %s: solved %d/%d in time %f"%(name,numFeasibleSolved+numInfeasibleSolved,len(self.testSet),feasibleTime+infeasibleTime)
        numFeasible = max(numFeasible,1)
        numInfeasible = max(numInfeasible,1)
        print "  Average time for feasible %f, infeasible %f"%(feasibleTime/numFeasible,infeasibleTime/numInfeasible)
        if numCostsCounted > 0: print "  Average cost suboptimality",costSuboptimality / numCostsCounted

    def testRaw(self,numRestarts=1):
        """Example: tests the performance of a numRestarts-random restart technique."""
        solverParams = copy.copy(self.groundTruthSolveParams)
        solverParams.startRandom = True
        solverParams.numRestarts = numRestarts
        self.test((lambda problem:problem.solve(self.robot,solverParams)),'%d random-restart'%(numRestarts,))


class IKDBTester(IKTestSet):
    """A helper for testing an IKDatabase approach."""
    def __init__(self,robot):
        IKTestSet.__init__(self,robot)
        #parameters for solving from scratch
        self.solveRawParams = IKSolverParams()
        self.solveRawParams.numRestarts = 100
        self.solveRawParams.localMethod = 'auto'
        #Used for DIRECT.  didn't work very well (5.28s, 231/1000 solved)
        #self.solveRawParams.numRestarts = 6000
        #self.solveRawParams.numIters = 100
        #self.solveRawParams.globalMethod = 'DIRECT'
        #Used for scipy.differential_evolution. 
        #self.solveRawParams.globalMethod = 'scipy.differential_evolution'
        #parameters for solving from adaptation
        self.solveAdaptParams = IKSolverParams()
        self.solveAdaptParams.localMethod = None
        #self.solveAdaptParams.localMethod = 'auto'

        self.db = IKDatabase(self.robot)

    def setNumIKSolveIters(self,iters):
        self.solveRawParams.numIters = iters
        self.solveAdaptParams.numIters = iters
        self.groundTruthSolveParams.numIters = numIters

    def generateDB(self,numTraining,numRestarts=100):
        print "Generating",numTraining,"training problems, with",numRestarts,"random restarts"
        trainingSet = [self.db.sampleRandomProblem() for i in range(numTraining)]
        print "**** Training ***"
        solverParams = copy.copy(self.solveRawParams)
        solverParams.numRestarts = numRestarts
        self.db.generate(trainingSet,solverParams=solverParams)

    def loadDB(self,fn):
        print "Loading database metadata from",fn+'.meta'
        self.db.loadMetadata(fn+'.meta')
        print "Loading database from",fn
        self.db.load(fn)
        print "IK database has",len(self.db.solutions),"instances"
        params = copy.copy(self.solveAdaptParams)
        params.localMethod = 'auto'
        #the following code is for debugging incorrectly built DBs
        """
        for i in xrange(len(self.db.solutions)):
            if self.db.solutions[i] is not None:
                p = self.db.getProblem(i)
                s = self.db.solutions[i]
                self.db.robot.setConfig(s)
                res = self.db.solveAdapt(p,params)
                if res is not None:
                    if p.costFunction(res) < p.costFunction(s):
                        print "DB solution was suboptimal by amount",p.costFunction(s)-p.costFunction(res)
                    self.db.solutions[i] = s
                else:
                    print "DB problem couldn't be solved?"
                print "  ",p.toJson()
                print "  Solution",s
                self.robot.setConfig(s)
                print "  Residual",p.constraintResidual(self.robot)
                raw_input()
        """
        
    def saveDB(self,fn):
        print "Saving IK database to",fn
        self.db.save(fn)
        print "Saving IK database metadata to",fn+'.meta'
        self.db.saveMetadata(fn+'.meta')

    def loadTest(self,fn):
        """Loads test instances with labels to a file -- packed version"""
        #return self.loadTestFeatures(fn,self.db.ikTemplate,self.db.featureNames)
        return IKTestSet.loadTest(self,fn)
        
    def saveTest(self,fn):
        """Saves test instances with labels to a file"""
        #return self.saveTestFeatures(fn,self.db.featureNames)
        return IKTestSet.saveTest(self,fn)

    def generateTest(self,numTesting):
        print "Generating",numTesting,"testing problems"
        problems = [self.db.sampleRandomProblem() for i in range(numTesting)]
        self.generateTestLabels(problems)

    def testDBPrediction(self,numQueries=1,recallBias=0.5):
        tp,tn,fp,fn=0,0,0,0
        predictionTime=0
        for p in self.testSet:
            t0 = time.time()
            res,confidence = self.db.predictFeasible(p.problem,numQueries,recallBias)
            t1 = time.time()
            predictionTime += t1-t0
            if res == True and p.feasible == True:
                tp += 1
            elif res == True and p.feasible == False:
                fp += 1
            elif res == False and p.feasible == True:
                fn += 1
            elif res == False and p.feasible == False:
                tn += 1
            else:
                raise RuntimeError("Ground truth labels aren't available...")
        print "Feasibility prediction %d-NN: precision %f, recall %f"%(numQueries,float(tp)/max(1,fp+tp),float(tp)/max(1,fn+tp))
        print "  Prediction time",predictionTime

    def testDB(self,numQueries=1):
        self.db.resetStats()
        self.test((lambda problem:self.db.solve(problem,numSolutionsToTry=numQueries,params=self.solveAdaptParams)[0]),'%d-NN query'%(numQueries,))
        print "  lookup time: %f, solve time: %f"%(self.db.lookupTime,self.db.solveTime)


class MultiIKDatabase:
    """An automatic manager of all IK database types.

    Attributes:
    - robot: the RobotModel used for solving
    - policy: the policy used during solve() (see below)
    - solveRawParams: the solver parameters used during raw solves (default
      100 restarts)
    - solveAdaptParams: the solver parameters used during adapt solves (default
      1 restart)
    - numSolutionsToTry: the number of prior solutions attempted during
      adaptation solves.
    - databases: a dictionary of structure keys to IKDatabase's.
    - databaseFiles: a dictionary of structure keys to filenames in which
      the database is stored.
    - size: the total number of primitives.
    - numDbLookups / dbLookupTime: stats about database lookups.
    - numAdaptSolves / numAdaptSolvesSuccessful / adaptSolveTime: stats about
      adaptation solves.
    - numRawSolves / numRawSolvesSuccessful / rawSolveTime: stats about
      adaptation solves.

    When solve() is called, a raw solver and/or an adaptation solver may be
    called depending on the value of the attribute 'policy'.  'policy' can be
    one of, or a list of, the following flags.
    - POLICY_RAW: a raw solve is called
    - POLICY_ADAPT: an adapt (local solve from a selected example) is called
    - POLICY_CURRENT: a local solve from the current configuration is called
    - POLICY_CURRENT_OR_ADAPT: a local solve from either a selected example, or
      from the current configuration, is called, depending on predicted
      quality.
    - POLICY_PREDICT: if the database predicts that the problem is feasible,
      solving continues.
    """
    POLICY_RAW = 0
    POLICY_ADAPT = 1
    POLICY_CURRENT = 2
    POLICY_PREDICT = 3
    POLICY_CURRENT_OR_ADAPT = 4

    def __init__(self,robot):
        self.robot = robot
        #the default solve policy
        self.policy = [MultiIKDatabase.POLICY_ADAPT]
        #parameters for solving from scratch
        self.solveRawParams = IKSolverParams()
        self.solveRawParams.numRestarts = 100
        self.solveRawParams.localMethod = 'auto'
        #parameters for solving from adaptation
        self.solveAdaptParams = IKSolverParams()
        self.solveAdaptParams.localMethod = 'auto'
        #parameters for solving from current
        self.solveCurrentParams = IKSolverParams()
        self.solveCurrentParams.localMethod = 'auto'
        #number of previous solutions to adapt
        self.numSolutionsToTry = 5
        #prediction infeasibility confidence
        self.infeasibilityConfidence = 0.98

        self.databases = dict()
        self.databaseFiles = dict()

        #number of solutions in database
        self.size = 0
        #number/time for looking up database
        self.numDbLookups = 0
        self.dbLookupTime = 0
        #number/time of adaptation solves
        self.numAdaptSolves = 0
        self.numAdaptSolvesSuccessful = 0
        self.adaptSolveTime = 0 
        #number/time of raw solves
        self.numRawSolves = 0
        self.numRawSolvesSuccessful = 0
        self.rawSolveTime = 0
        
    def stats(self):
        return {'size':self.size,
                'numDbLookups':self.numDbLookups,'dbLookupTime':self.dbLookupTime,
                'numAdaptSolves':self.numAdaptSolves,'numAdaptSolvesSuccessful':self.numAdaptSolvesSuccessful,'adaptSolveTime':self.adaptSolveTime,
                'numRawSolves':self.numRawSolves,'numRawSolvesSuccessful':self.numRawSolvesSuccessful,'rawSolveTime':self.rawSolveTime}

    def printStats(self):
        print "MultiIKDatabase:"
        print "  Problems stored:",self.size
        print "  Databases:",len(self.databases)
        print "  Database lookups: %d, time %f"%(self.numDbLookups,self.dbLookupTime)
        print "  Problems solved via adaptation: %d, %d successful, time %f"%(self.numAdaptSolves,self.numAdaptSolvesSuccessful,self.adaptSolvetime)
        print "  Problems solved via raw solver: %d, %d successful, time %f"%(self.numRawSolves,self.numRawSolvesSuccessful,self.rawSolvetime)

        for k in self.databases:
            print "  IKDatabase",k,":"
            print "    File:",self.databaseFiles[k]
            print "    Num problems:",self.databases[k].numProblems()
            print "    Num queries:",self.databases[k].numQueries
            print "    Lookup time:",self.databases[k].lookupTime
            print "    Solve time:",self.databases[k].solveTime

    def save(self,folder_prefix):
        """Saves the entire package to disk."""
        self.savePackageData(folder_prefix)
        for k in self.databases:
            self.saveDatabase(k,folder_prefix)

    def load(self,folder_prefix):
        """Loads the entire package from disk."""
        self.loadPackageData(folder_prefix)
        for k in self.databaseFiles:
            #already loaded metadata, don't need to do it again
            #self.loadDatabase(k,folder_prefix)
            db = self.databases[k]
            assert hasattr(db,'fileName')
            fn = os.path.join(folder_prefix,db.fileName)
            db.load(fn)

    def savePackageData(self,folder_prefix):
        """Saves the package data to folder_prefix/package.txt. Called during
        save()/saveDatabase() calls."""
        import json
        package = {}
        package['type'] = "MultiIKDatabase"
        package['robot'] = self.robot.getName()
        package['size'] = self.size
        package['stats'] = self.stats()
        package['branches'] = []
        for (k,v) in self.databases.iteritems():
            if k not in self.databaseFiles:
                v.fileName = self.databaseFiles[k] = 'db'+str(len(self.databaseFiles))+'.txt'
            info = {'template':v.ikTemplate,'dbfile':v.fileName,'metafile':self.databaseFiles[k]+'.meta'}
            package['branches'].append(info)
        mkdir_p(folder_prefix)
        with open(os.path.join(folder_prefix,"package.txt"),'w') as f:
            json.dump(package,f)
        print "Saving package data to",os.path.join(folder_prefix,"package.txt")
        #need to save metadata
        for (k,v) in self.databases.iteritems():
            fnmeta = os.path.join(folder_prefix,v.fileName+'.meta')
            print "  Saving IK database meta data to",fnmeta
            v.saveMetadata(fnmeta)

    def loadPackageData(self,folder_prefix):
        """Loads the package data from folder_prefix/package.txt."""
        import json
        with open(os.path.join(folder_prefix,"package.txt"),'r') as f:
            obj = json_byteify(json.load(f))
            if obj['type'] != "MultiIKDatabase":
                raise IOError("Invalid JSON object loaded from package.txt")
            package = obj
            if package['robot'] != self.robot.getName():
                print "MultiIKDatabase: warning, robot in package does not match that of robot provided to constructor"
            self.size = int(package['size'])
            #package['stats'] = self.stats()
            self.databases = dict()
            self.databaseFiles = dict()
            for b in package['branches']:
                template = b['template']
                dbfile = b['dbfile']
                metafile = b['metafile']
                k = features.structure(template,hashable=True)
                self.databases[k] = IKDatabase(self.robot)
                self.databases[k].ikTemplate = template
                self.databaseFiles[k] = dbfile
                self.databases[k].fileName = dbfile
                self.databases[k].savedProblemCount = 0
                fnmeta = os.path.join(folder_prefix,metafile)
                self.databases[k].loadMetadata(fnmeta)
        return

    def saveDatabase(self,k,folder_prefix):
        """Saves the database indexed by key k.  If k wasn't in the package, the
        package data is saved again too.  Called during save() calls."""
        db = self.databases[k]
        self.assignDatabaseFilename(db,k)
        fn = os.path.join(folder_prefix,db.fileName)
        fnmeta = os.path.join(folder_prefix,db.fileName+'.meta')
        db.save(fn)
        db.saveMetadata(fnmeta)

    def loadDatabase(self,k,folder_prefix):
        """Loads the database indexed by key k."""
        db = self.databases[k]
        assert hasattr(db,'fileName')
        fn = os.path.join(folder_prefix,db.fileName)
        fnmeta = os.path.join(folder_prefix,db.fileName+'.meta')
        db.loadMetadata(fnmeta)
        db.load(fn)

    def assignDatabaseFilename(self,db,k=None):
        if k is None: k = self.getDatabaseKey(db)
        if k not in self.databaseFiles:
            #k wasnt in the package
            db.fileName = self.databaseFiles[k] = 'db'+str(len(self.databaseFiles))+'.txt'
            return True
        return False

    def getDatabaseKey(self,db):
        """Returns the key associated with the given database"""
        #note: this is slower compared to purely getting the structure of the
        #IK template if there are a LOT of databases...
        for (k,v) in self.databases.iteritems():
            if v is db:
                return k
        return None
        
    def solve(self,problem,policy=None):
        """Solve a new IKProblem, or one or more IKObjectives.
        Manages the database as necessary to populate it incrementally.
        """
        if isinstance(problem,IKObjective):
            problem = IKProblem(problem)
        elif isinstance(problem,(list,tuple)):
            problem = IKProblem(*problem)
        assert isinstance(problem,IKProblem),"Argument must be an IKProblem or one or more IKObjectives"
        db = self.getDatabase(problem)
        return self.solveWithDatabase(db,problem,policy)

    def getDatabase(self,problem):
        """Returns the IKDatabase associated with the given problem.
        Creates the database, if it doesn't already exist."""
        self.numDbLookups += 1
        t0 = time.time()
        obj = problem.toJson()
        s = features.structure(obj,hashable=True)
        try:
            db = self.databases[s]
        except KeyError:
            #a new database
            db = IKDatabase(self.robot)
            db.ikTemplate = problem.toJson()
            db.savedProblemCount = 0
            db.featureNames = []
            self.databases[s] = db
        t1 = time.time()
        self.dbLookupTime += (t1-t0)
        return db

    def solveWithDatabase(self,db,problem,policy=None):
        """A slightly faster version of solve() when an internal database has
        been identified.  If you are solving many problems with the same
        structure but with differing parameters, you can avoid some overhead
        using getDatabase() and multiple calls to solveWithDatabase(),
        rather than multiple calls to solve().

        Tests on a 2011 laptop indicate the overhead is approximately 100
        microseconds.
        """
        if policy is None: policy = self.policy
        if not isinstance(policy,(list,tuple)):
            policy = [policy]
        features = None
        for p in policy:
            if p == MultiIKDatabase.POLICY_CURRENT:
                result = problem.solve(self.robot,self.solveCurrentParams)
                if result is not None:
                    self.onSolveSuccess(p,db,problem,result)
                    return result
                else:
                    self.onSolveFailure(p,db,problem)
            elif p == MultiIKDatabase.POLICY_RAW:
                result = self.solveRaw(db,problem)
                #this already calls onSolveSuccess/Failure, don't need to do it again
                if result is not None:
                    return result
            elif p == MultiIKDatabase.POLICY_ADAPT or p == MultiIKDatabase.POLICY_CURRENT_OR_ADAPT:
                #how many previous solutions to try to adapt
                k = min(self.numSolutionsToTry,db.numProblems())
                if len(db.featureNames)==0:
                    #no features available
                    k=0
                if k == 0:
                    #empty database
                    if p == MultiIKDatabase.POLICY_CURRENT_OR_ADAPT:
                        #solve from current
                        result = problem.solve(self.robot,self.solveCurrentParams)
                        if result is not None:
                            self.onSolveSuccess(p,db,problem,(result,-1))
                            return result
                        else:
                            self.onSolveFailure(p,db,problem)
                    continue
                #do the solve
                self.numAdaptSolves += 1
                t0 = time.time()
                features = np.array(db.problemToFeatures(problem))
                result,index = db.solve(problem,k,self.solveAdaptParams,features=features,tryCurrent=(p == MultiIKDatabase.POLICY_CURRENT_OR_ADAPT))
                t1 = time.time()
                self.adaptSolveTime += t1-t0
                if result is not None:
                    if index >= 0:
                        #best result: solved by primitive adaptation
                        self.numAdaptSolvesSuccessful += 1
                        self.onSolveSuccess(p,db,problem,(result,index))
                        #do we want to add this problem to the database?
                        #only if it's sufficiently far away
                        return result
                    else:
                        self.onSolveSuccess(p,db,problem,(result,index))
                        return result
                else:
                    self.onSolveFailure(p,db,problem)
            else:
                #POLICY_PREDICT
                #how many previous solutions to try to adapt
                k = min(self.numSolutionsToTry,db.numProblems())
                if k == 0:
                    #empty database
                    continue
                if features is None:
                    features = np.array(db.problemToFeatures(problem))
                prediction,confidence = db.predictFeasible(problem,k,features=features)
                if prediction or confidence < 0.65:
                    print "Predicted feasible or unconfident:",confidence
                    #predicted feasible or not confident, run global solver
                    self.onSolveSuccess(p,db,problem,prediction)
                elif confidence<self.infeasibilityConfidence:
                    print "Semi-confident that problem is infeasible:",confidence
                    #not entirely confident, so put on backburner
                    self.onSolveFailure(p,db,problem)
                    return None
                else:
                    print "Confident that problem is infeasible:",confidence
                    self.onSolveSuccess(p,db,problem,prediction)
                    return None
        return None

    def solveRaw(self,db,problem,log=True):
        """Solves a problem belonging to the right structure as the given
        database db using the solve raw method.  Calls onSolveSuccess/Failure
        depending on the result
        """
        t0 = time.time()
        res = db.solveRaw(problem,self.solveRawParams)
        t1 = time.time()
        if log:
            if res is not None:
                self.numRawSolvesSuccessful += 1
            self.numRawSolves += 1
            self.rawSolveTime += t1-t0

        if res is not None:
            self.onSolveSuccess(MultiIKDatabase.POLICY_RAW,db,problem,res)
        else:
            self.onSolveFailure(MultiIKDatabase.POLICY_RAW,db,problem)
        return res

    def onSolveSuccess(self,policy,db,problem,solution):
        """Called whenever a problem is solved for a given DB.
        The policy type is given in policy.  Note that policy may be
        POLICY_PREDICT if the prediction step was confident."""
        return

    def onSolveFailure(self,policy,db,problem):
        """Called whenever a problem is not solved for a given DB.
        The attempted policy type is given in policy. Note that policy may be
        POLICY_PREDICT if the prediction step was unconfident."""
        return
    
    def predictFeasible(self,problem,numSolutionsToQuery,recallBias=0.5):
        db = self.getDatabase(problem)
        return db.predictFeasible(problem,numSolutionsToQuery,recallBias)

    def sampleRandomProblem(self,featureExpandAmountRel=0,featureExpandAmountAbs=0):
        if len(self.databases)==0: return None
        keys = self.databases.keys()
        weights = [self.databases[k].numQueries for k in keys]
        k = sample_weighted(weights,keys)
        return self.databases[k].sampleRandomProblem(featureExpandAmountRel,featureExpandAmountAbs)

    def add(self,db,problem,solution):
        """Adds a problem to the indicated database."""
        self.size += 1

        db.solutions.append(solution)
        if len(db.featureNames) > 0:
            db.problemFeatures.append(db.problemToFeatures(problem))

            #NN data structure is appromately O(n log n) time to build...
            #how often do we want to re-build it?
            if not hasattr(db,'nnBuildSize') or db.numProblems() > 10 + db.nnBuildSize:
                db.buildNNDataStructure()
                db.clearFeasiblePredictionLearning()
        else:
            obj = problem.toJson()
            temp = IKProblem()
            temp.fromJson(obj)
            db.problems.append(temp)

    def metricLearn(self,numIters):
        for db in self.databases.itervalues():
            db.metricLearn(numIters)
            db.buildNNDataStructure()
            #db.clearFeasiblePredictionLearning()

class ManagedIKDatabase(MultiIKDatabase):
    """An automatic manager of all IK database types.  Also has functionality
    for background generation of IK databases.

    Important: if you care about the robot model's configuration, you will need
    to create a temporary model to be passed to this database.  Otherwise, the
    background thread will modify the configuration.

    Important: all non-constant parameters used by the feasibility test /
    cost function in a problem must be provided as arguments to the
    IKProblem.setFeasibilityTest / setCostFunction methods!  In other words,
    the IKProblem needs to *own* all variables that affect the solution.

    Usage:
    functionfactory.registerDefaultFunctions()
    functionfactory.registerCollisionFunction(world):
    db = ManagedIKDatabase(world.robot(0),[folder])
    #optional: start the background thread.  It will automatically populate
    #the database as you proceed.
    db.startBackgroundLoop()
    while (True):
        problem = make a new IK problem (either IKProblem instance, or list of IKObjectives)
        soln = db.solve(problem)
        if soln is not None:
            print "Got a solution"
    #this is not strictly necessary, but you may want to do it just to be nice...
    db.stopBackgroundLoop()
    """
    def __init__(self,robot,folder_prefix=None):
        MultiIKDatabase.__init__(self,robot)
        self.policy = [MultiIKDatabase.POLICY_CURRENT_OR_ADAPT,
                       MultiIKDatabase.POLICY_PREDICT,
                       MultiIKDatabase.POLICY_RAW]
        if folder_prefix is None:
            folder_prefix = robot.getName()
        self.folder_prefix = folder_prefix
        #maximum number of solutions in database
        self.maxSize = 10000000

        self.featureMiner = features.IncrementalMultiStructureFeatureMiner()
        #queue of un-attempted problems or examples of questionable quality
        self.backburner = []
        #time for doing random package management tasks
        self.packageManagementTime = 0
        #number/time of auto-populated 
        self.numAutoPopulated = 0
        self.numAutoPopulatedSuccessful = 0
        self.autoPopulateTime = 0

        #dirty flags
        self.databaseMetadataChanged = dict()
        self.databaseSavedProblemCount = dict()
        self.packageChanged = False
        self.backburnerChanged = False
        self.backburnerEnabled = True

        #lock that's used around all calls that might modify the database
        import threading
        self.lock = threading.RLock()

        #load on creation
        if os.path.isdir(folder_prefix):
            try:
                self.load()
            except IOError as e:
                print "Warning, couldn't load managed IK database from",folder_prefix
                c = raw_input("Folder exists, do you want to override it? (y/n) > ")
                if c != 'y':
                    raise IOError("IK database creation canceled")
        else:
            print "IK database will be placed in",folder_prefix
        
    def stats(self):
        return {'size':self.size,
                'packageManagementTime':self.packageManagementTime,
                'numDbLookups':self.numDbLookups,'dbLookupTime':self.dbLookupTime,
                'numAdaptSolves':self.numAdaptSolves,'numAdaptSolvesSuccessful':self.numAdaptSolvesSuccessful,'adaptSolveTime':self.adaptSolveTime,
                'numRawSolves':self.numRawSolves,'numRawSolvesSuccessful':self.numRawSolvesSuccessful,'rawSolveTime':self.rawSolveTime,
                'numAutoPopulated':self.numAutoPopulated,'numAutoPopulatedSuccessful':self.numAutoPopulatedSuccessful,'autoPopulateTime':self.autoPopulateTime}

    def printStats(self):
        print "ManagedIKDatabase:"
        print "  Problems stored:",self.size
        print "  Databases:",len(self.databases)
        print "  Database lookups: %d, time %f"%(self.numDbLookups,self.dbLookupTime)
        print "  Problems solved via adaptation: %d, %d successful, time %f"%(self.numAdaptSolves,self.numAdaptSolvesSuccessful,self.adaptSolvetime)
        print "  Problems solved via raw solver: %d, %d successful, time %f"%(self.numRawSolves,self.numRawSolvesSuccessful,self.rawSolvetime)

        print "  Problems on backburner queue:",len(self.backburner)
        print "  Problems solved in background: %d, %d successful, time %f"%(self.numAutoPopulated,self.numAutoPopulatedSuccessful,self.autoPopulateTime)
        for k in self.databases:
            print "  IKDatabase",k,":"
            print "    File:",self.databaseFiles[k]
            print "    Num problems:",len(self.databases[k].numProblems())
            print "    Num queries:",self.databases[k].numQueries
            print "    Lookup time:",self.databases[k].lookupTime
            print "    Solve time:",self.databases[k].solveTime

    def flush(self):
        """Flushes all changes to the database to disk."""
        with self.lock:
            print "Flushing changes to disk:"
            if self.packageChanged:
                print "  Package changed"
            if self.backburnerChanged:
                print "  Backburner changed"
            numMetadataChanged = 0
            for (k,v) in self.databaseMetadataChanged.iteritems():
                if v:
                    numMetadataChanged += 1
                    print "  Metadata for",features.structure(self.databases[k].ikTemplate,hashable=False),"changed"
            for (k,db) in self.databases.iteritems():
                cnt = self.databaseSavedProblemCount.get(k,0)
                if cnt < db.numProblems():
                    print "  %d new problems for"%(db.numProblems()-cnt,),features.structure(db.ikTemplate,hashable=False)
            #now do the saving
            if self.packageChanged or self.backburnerChanged or numMetadataChanged > 0:
                self.savePackageData()
            for (k,db) in self.databases.iteritems():
                cnt = self.databaseSavedProblemCount.get(k,0)
                if cnt < db.numProblems():
                    self.saveDatabaseIncrement(db)


    def save(self):
        with self.lock:
            self.savePackageData()
            for k in self.databases:
                self.saveDatabase(k)
            self.databaseMetadataChanged = dict()
            self.packageChanged = False
            self.backburnerChanged = False

    def load(self):
        """Loads the entire package from disk.  Does not need to be called
        by user."""
        self.loadPackageData()
        self.databaseMetadataChanged = dict()
        self.databaseSavedProblemCount = dict()
        self.packageChanged = False
        self.backburnerChanged = False
        for k in self.databases:
            self.loadDatabase(k)

        #rebuild feature miners
        allproblems = []
        for (k,v) in self.databases.iteritems():
            for i in xrange(v.numProblems()):
                allproblems.append(v.getProblem(i).toJson())
        self.featureMiner = features.IncrementalMultiStructureFeatureMiner(dataset=allproblems)
        #clear new feature counters
        for k,v in self.featureMiner.structureToFeatureMiner.iteritems():
            if k not in self.databases:
                print "Uh... structure",k,"was discovered by feature miner"
                print "Available keys:"
                print self.databases.keys()
                raw_input()
            for f in self.databases[k].featureNames:
                if f not in v.features:
                    print "Warning: feature",f,"discovered before was not in dynamically discovered features",v.features
                    raw_input()
            v.newFeatureCount = 0

    def savePackageData(self):
        """Saves the package data to folder_prefix/package.txt. Called during
        save()/saveDatabase() calls, and called automatically if persistent
        is True."""
        MultiIKDatabase.savePackageData(self,self.folder_prefix)
        self.databaseMetadataChanged = dict()
        self.packageChanged = False
        self.backburnerChanged = False
        import json
        with open(os.path.join(self.folder_prefix,"backburner.txt"),'w') as f:
            for p,s in self.backburner:
                example = {'problem':p,'solution':s}
                json.dump(example,f)
                f.write('\n')

    def loadPackageData(self):
        """Loads the package data from folder_prefix/package.txt."""
        MultiIKDatabase.loadPackageData(self,self.folder_prefix)
        self.databaseMetadataChanged = dict()
        self.packageChanged = False
        self.backburnerChanged = False
        import json
        self.backburner = []
        with open(os.path.join(self.folder_prefix,"backburner.txt"),'r') as f:
            for line in f.readlines():
                obj = json.loads(line)
                self.backburner.append((obj['problem'],obj['solution']))
        return

    def saveDatabase(self,k):
        """Saves the database indexed by key k.  If k wasn't in the package, the
        package data is saved again too.  Called during save() calls, and
        called automatically if persistent is True."""
        MultiIKDatabase.saveDatabase(self,k,self.folder_prefix)
        self.databaseMetadataChanged[k] = False
        self.databaseSavedProblemCount[k] = self.databases[k].numProblems()

    def loadDatabase(self,k):
        """Loads the database indexed by key k."""
        MultiIKDatabase.loadDatabase(self,k,self.folder_prefix)
        self.databaseMetadataChanged[k] = False
        self.databaseSavedProblemCount[k] = self.databases[k].numProblems()

    def saveDatabaseIncrement(self,db):
        """Saves the latest numLatest entries to database k"""
        self.assignDatabaseFilename(self,db)
        fn = os.path.join(self.folder_prefix,db.fileName)
        print "Save increment: saving increment to",fn
        assert len(db.problems) == 0,"Can't save an increment without a feature space..." 
        with open(fn,'a') as f:
            k = self.getDatabaseKey(db)
            start = self.databaseSavedProblemCount.get(k,0)
            for i,s in enumerate(db.solutions[start:]):
                row = start+i
                assert row < len(db.problemFeatures)
                f.write(' '.join(str(v) for v in db.problemFeatures[row])+'\n')
                if s is None:
                    f.write('\n')
                else:
                    f.write(' '.join(str(v) for v in s)+'\n')
            self.databaseSavedProblemCount[k] = len(db.solutions)        

    def solveWithDatabase(self,db,problem,policy=None):
        with self.lock:
            return MultiIKDatabase.solveWithDatabase(self,db,problem,policy)

    def getDatabase(self,problem):
        with self.lock:
            return MultiIKDatabase.getDatabase(self,problem)

    def add(self,db,problem,solution):
        #print "ManagedIKDatabase: add",solution
        with self.lock:
            MultiIKDatabase.add(self,db,problem,solution)
            #determine if the features have changed
            t0 = time.time()
            obj = problem.toJson()
            miner = self.featureMiner.add(obj)
            if miner.newFeatureCount > 0: #there are some new features, update the database
                fs = miner.getFeatureList()
                ts = miner.getFeatureTypes()
                newFeatureNames = db.featureNames[:]
                for (f,t) in zip(fs[-miner.newFeatureCount:],ts[-miner.newFeatureCount:]):
                    if t == 'integer' or t == 'real':
                        newFeatureNames.append(f)
                    else: #t == 'string' or t == 'mixed'
                        #TODO: split databases by string type
                        pass
                #regenerate features for the given database
                print "Regenerating features for database",features.structure(obj,hashable=False)
                print "  Feature names:",db.featureNames,"mapping to",newFeatureNames
                db.setIKProblemSpace(db.ikTemplate,newFeatureNames,None)
                db.autoSetFeatureRanges()
                self.databaseMetadataChanged[features.structure(obj,hashable=True)] = True
                miner.newFeatureCount = 0
                t1 = time.time()
                self.packageManagementTime += t1-t0

            if db.featureRanges is None and len(db.featureNames) > 0 and solution is not None:
                print "Got a first feasible solution..."
                #first feasible solution
                db.autoSetFeatureRanges()

            if db.featureRanges is not None and solution is not None:
                t0 = time.time()
                featureList = db.problemToFeatures(problem)
                changed = False
                for i,((a,b),f) in enumerate(zip(db.featureRanges,featureList)):
                    if f < a:
                        db.featureRanges[i] = (f,db.featureRanges[i][1])
                        changed = True
                    elif f > b:
                        db.featureRanges[i] = (db.featureRanges[i][0],f)
                        changed = True
                if changed:
                    k = features.structure(obj,hashable=True)
                    self.databaseMetadataChanged[k] = True
                t1 = time.time()
                self.packageManagementTime += t1-t0
        return

    def onSolveFailure(self,policy,db,problem):
        if policy == MultiIKDatabase.POLICY_RAW:
            print "IKDB: Raw solve fail"
            if self.size < self.maxSize:
                self.add(db,problem,None)
        elif policy == MultiIKDatabase.POLICY_PREDICT:
            if self.backburnerEnabled:
                with self.lock:
                    print "IKDB: Adding to backburner..."
                    if len(self.backburner) > 100:
                        del self.backburner[random.randint(0,len(self.backburner)-1)]
                    self.backburner.append((problem.toJson(),None))
                    self.backburnerChanged = True

    def onSolveSuccess(self,policy,db,problem,solution):
        if policy == MultiIKDatabase.POLICY_CURRENT or (policy == MultiIKDatabase.POLICY_CURRENT_OR_ADAPT and solution[1]==-1):
            #add to backburner?
            if self.backburnerEnabled:
                with self.lock:
                    #print "Solved from current, adding to backburner..."
                    if len(self.backburner) > 100:
                        del self.backburner[random.randint(0,len(self.backburner)-1)]
                    self.backburner.append((problem.toJson(),solution[0]))
                    self.backburnerChanged = True
        elif policy == MultiIKDatabase.POLICY_RAW:
            print "IKDB: Raw solve success"
            if self.size < self.maxSize:
                self.add(db,problem,solution)
        elif (policy == MultiIKDatabase.POLICY_ADAPT or policy == MultiIKDatabase.POLICY_CURRENT_OR_ADAPT) and solution[1] > 0:
            #a non-closest example was adapted
            if self.size < self.maxSize:
                print "Non-closest problem adapted:",solution[1],"adding to db"
                self.add(db,problem,solution[0])

    def startBackgroundLoop(self,saveRate=100):
        """Starts a thread that auto-populates the database.  Stops when
        stopBackgroundLoop is called, or the main thread exits.

        It is important that no other threads change the database while it is
        locked.  
        
        Every saveRate steps, a flush() is called to save all changes to
        the disk.  If saveRate == 0, no saves are called.
        """
        import threading
        if hasattr(self,'thread'):
            raise RuntimeError("Can't run background thread twice (Python doesn't do multiprocessing)")
        def run_in_background(db):
            numSteps = 0
            while not db.killBackgroundThread:
                t0 = time.time()
                if db.backgroundStep():
                    t1 = time.time()
                    #print "sleep",t1-t0+0.1
                    #time.sleep(t1-t0+0.1)
                    time.sleep(max(0.1-(t1-t0),0))
                else:
                    t1 = time.time()
                    #print "sleep",t1-t0+0.1
                    #time.sleep(t1-t0+0.1)
                    time.sleep(max(0.1-(t1-t0),0))
                numSteps += 1
                if saveRate > 0 and numSteps % saveRate == 0:
                    db.flush()
        self.killBackgroundThread = False
        self.thread = threading.Thread(target=run_in_background,args=(self,))
        self.thread.daemon=True
        self.thread.start()

    def stopBackgroundLoop(self):
        """Stops the thread that auto-populates the database.  The thread is
        also terminated when the main thread exits."""
        if not hasattr(self,'thread'):
            return
        self.killBackgroundThread = True
        self.thread.join()
        delattr(self,'killBackgroundThread')
        delattr(self,'thread')

    def populate(self,count=None):
        """Performs self-population of this database, and stops when it
        is filled. If count is provided, then it only performs at most count
        steps of self-population.

        Warning, if Ctrl+C is pressed while the database is saving,
        it might become corrupted.
        """
        nsteps = 0
        while True:
            nsteps += 1
            if self.backgroundStep()==False:
                break
            if count is not None and nsteps >= count:
                break

    def backgroundStep(self):
        """Processes more items for the database in the background.
        Returns False if there's no work to be done."""
        if self.size >= self.maxSize:
            return False
        with self.lock:
            if len(self.backburner) > 0:
                problemJson,solution = self.backburner.pop(-1)
                problem = IKProblem()
                problem.fromJson(problemJson)
                self.backburnerChanged = True
                self.backburnerEnabled = False
                
                db = self.getDatabase(problem)
                #try to solve with database
                k = min(self.numSolutionsToTry,db.numProblems())
                prediction,confidence = db.predictFeasible(problem,k)
                if prediction == False and confidence >= self.infeasibilityConfidence:
                    #confident it's infeasible, can just quit
                    return True
                t0 = time.time()
                #don't solve from current
                solved = self.solveWithDatabase(db,problem,policy=[MultiIKDatabase.POLICY_ADAPT,MultiIKDatabase.POLICY_RAW])
                t1 = time.time()
                self.numAutoPopulated += 1
                self.autoPopulateTime += t1-t0
                if solved is not None:
                    if solution is not None:
                        self.robot.setConfig(solution)
                        sold = problem.score(self.robot)
                        self.robot.setConfig(solved)
                        snew = problem.score(self.robot)
                        if sold < snew:
                            #previous solution was better than database
                            print "Old solution was better than database, adding"
                            self.add(db,problem,solution)
                    self.numAutoPopulatedSuccessful += 1
                self.backburnerEnabled = True
                return True
            else:
                #generate a new problem
                if len(self.databases)==0: return False
                keys = self.databases.keys()
                weights = [self.databases[k].numQueries for k in keys]
                k = sample_weighted(weights,keys)
                db = self.databases[k]
                if db.featureRanges is None:
                    t0 = time.time()
                    db.autoSetFeatureRanges()
                    t1 = time.time()
                    self.packageManagementTime += t1-t0
                    #no feasible problems?  need at least 1 to start generating 
                    if db.featureRanges is None:
                        return True
                    self.databaseMetadataChanged[k] = True
                problem = db.sampleRandomProblem(0.1,0.1)
                t0 = time.time()
                solved = self.solveRaw(db,problem,log=False)
                t1 = time.time()
                self.numAutoPopulated += 1
                self.autoPopulateTime += t1-t0
                if solved:
                    self.numAutoPopulatedSuccessful += 1
        return True


class MultiIKDBTester (IKTestSet):
    def __init__(self,robot):
        IKTestSet.__init__(self,robot)
        self.db = MultiIKDatabase(self.robot)
        self.db.persistent = False
        self.db.policy = [MultiIKDatabase.POLICY_ADAPT]

    def setNumIKSolveIters(self,iters):
        self.groundTruthSolveParams.numIters = iters
        self.db.solveRawParams.numIters = iters
        self.db.solveAdaptParams.numIters = iters

    def loadDB(self,prefix):
        return self.db.load(prefix)

    def saveDB(self,prefix):
        return self.db.save(prefix)

    def generateDB(self,numTraining):
        print "Generating",numTraining,"training problems"
        print "**** Training ***"
        for iter in xrange(numTraining):
            p = self.db.sampleRandomProblem(0.1,0.1)
            db = self.db.getDatabase(p)
            s = self.db.solveRaw(db,p)
            self.db.add(db,p,s)
    
    def generateTest(self,numTesting):
        print "Generating",numTesting,"testing problems"
        problems = [self.db.sampleRandomProblem(0.1,0.1) for i in range(numTesting)]
        self.generateTestLabels(problems)


    def testDB(self,numQueries=1):
        oldNumQueries = self.db.numSolutionsToTry
        self.db.numSolutionsToTry = numQueries
        self.test((lambda problem:self.db.solve(problem)),'%d-NN query'%(numQueries,))
        self.db.numSolutionsToTry = numQueries

    def testDBPrediction(self,numQueries=1,recallBias=0.5):
        tp,tn,fp,fn=0,0,0,0
        predictionTime=0
        for p in self.testSet:
            t0 = time.time()
            db = self.db.getDatabase(p.problem)
            res,confidence = db.predictFeasible(p.problem,numQueries,recallBias)
            t1 = time.time()
            predictionTime += t1-t0
            if res == True and p.feasible == True:
                tp += 1
            elif res == True and p.feasible == False:
                fp += 1
            elif res == False and p.feasible == True:
                fn += 1
            elif res == False and p.feasible == False:
                tn += 1
            else:
                raise RuntimeError("Ground truth labels aren't available...")
        print "Feasibility prediction %d-NN: precision %f, recall %f"%(numQueries,float(tp)/max(1,fp+tp),float(tp)/max(1,fn+tp))
        print "  Prediction time",predictionTime




#begin klampt ik module API overloads
global _masters
_masters = {}

def _start_master(robot):
    global _masters
    if robot.getName() not in _masters:
        _masters[robot.getName()] = ManagedIKDatabase(robot)
        _masters[robot.getName()].startBackgroundLoop()

def objective(*args,**kwargs):
    """Duplicate of Klamp't ik module API"""
    return ik.objective(*args,**kwargs)

def fixed_objective(*args,**kwargs):
    """Duplicate of Klamp't ik module API"""
    return ik.fixed_objective(*args,**kwargs)

def objects(objectives):
    """Duplicate of Klamp't ik module API"""
    return ik.objects(objectives)    

def solver(objectives):
    """Duplicate of Klamp't ik module API"""
    return ik.solver(objectives)    

def residual(objectives):
    """Duplicate of Klamp't ik module API"""
    return ik.residual(objectives)

def jacobian(objectives):
    """Duplicate of Klamp't ik module API"""
    return ik.jacobian(objectives)

def solve_global(objectives,iters=1000,tol=1e-3,activeDofs=None,feasibilityCheck=None,costFunction=None):
    """Solves the IK problem given by the objectives.  Optionally can take a feasibility check
    and a cost function. 

    This will start / use an IK database for the given problem.

    feasibilityCheck and costFunction, if given, must be names of functions defined using the API in
    functionfactory.py.
    """
    global _masters
    if not hasattr(objectives,'__iter__'):
        objectives = [objectives]
    robot = objectives[0].robot
    _start_master(robot)
    p = IKProblem(*objectives)
    if activeDofs != None:
        p.setActiveDofs(activeDofs)
    if feasibilityCheck != None:
        p.setFeasibilityTest(feasibilityCheck,[])
    if costFunction != None:
        p.setCostFunction(costFunction,[])
    res = _masters[robot.getName()].solve(p)
    if res is not None:
        return res
    return None


def solve(objectives,iters=1000,tol=1e-3,activeDofs=None,feasibilityCheck=None,costFunction=None):
    """Solves the IK problem given by the objectives.  Optionally can take a feasibility check
    and a cost function. 

    This will start / use an IK database for the given problem.

    feasibilityCheck and costFunction, if given, must be names of functions defined using the API in
    functionfactory.py.
    """
    return solve_global(objectives,iters,tol,activeDofs,feasibilityCheck,costFunction)

def solve_nearby(objectives,maxDeviation,iters=1000,tol=1e-3,activeDofs=None,feasibilityCheck=None,costFunction=None):
    """Solves the local IK problem given by the objectives.  Optionally can take a feasibility check
    and a cost function. 

    This will start / use an IK database for the given problem.

    feasibilityCheck and costFunction, if given, must be names of functions defined using the API in
    functionfactory.py.
    """
    global _masters
    if not hasattr(objectives,'__iter__'):
        objectives = [objectives]
    robot = objectives[0].robot
    _start_master(robot)
    p = IKProblem(objectives)
    if activeDofs != None:
        p.setActiveDofs(activeDofs)
    if feasibilityCheck != None:
        p.setFeasibilityTest(feasibilityCheck,[])
    if costFunction != None:
        p.setCostFunction(costFunction,[])
    q = robot.getConfig()
    qmin,qmax = robot.getJointLimits()
    for d in range(robot.numLinks()):
        qmin[d] = max(qmin[d],q[d]-maxDeviation)
        qmax[d] = min(qmax[d],q[d]+maxDeviation)
    p.setJointLimits(qmin,qmax)

    res = _masters[robot].solve(p)
    if res is not None:
        assert len(res)==2
        return res[0]
    return None

def flush():
    """Forces a flush of the databases"""
    global _masters
    for (k,v) in _masters.iteritems():
        v.flush()


