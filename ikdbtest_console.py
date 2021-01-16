#Python 2/3 compatibility
from __future__ import print_function,division,absolute_import
from builtins import input,range
from six import iteritems

from ikdb import *
from ikdb import IKDBTester,MultiIKDBTester
from ikdb import functionfactory
from ikdb.utils import mkdir_p
from klampt import *
import pkg_resources
if pkg_resources.get_distribution('klampt').version >= '0.7':
    from klampt.model import ik
    from klampt.io import loader
else:
    from klampt import ik,loader
import numpy as np
import sys
import random
import math
import os

#preload to avoid later delays
import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors,BallTree
from scipy.optimize import differential_evolution

def make_ikdb_tester(robot,iktemplate,ikfeatures,featureranges,
    numTrain=1000,numTest=1000,
    trainfile='auto',testfile='auto',
    redoTrain=False,redoTest=False):
    """Creates an IKDBTester or MultiIKDBTester for testing
    the performance of an IK database.

    Arguments:
    - robot: the Klampt RobotModel being used
    - iktemplate: one or more IKProblems. If multiple templates are
      given, then a MultiIKDBTester is returned
      Otherwise, an IKDBTester is returned.
    - ikfeatures: one or more lists of features for each template.
    - featureranges: one or more lists of ranges for the features
      in the form [(a1,b1),...,(ak,bk)] where ai<=fi<=bi is the features
      range.
    - numTrain: # of training examples
    - numTest: # of testing examples
    - trainfile: an existing IK database or file to save to
    - testfile: an existing test set or file to save to
    - redoTrain: True if you wish to re-generate the training set 
      even if it exists.
    - redoTest: True if you wish to re-generate the testing set
      even if it exists.
    """
    multi = hasattr(iktemplate,'__iter__')
    if multi:
        assert len(iktemplate) == len(ikfeatures)
        assert len(iktemplate) == len(featureranges)
    if trainfile == 'auto':
        dbfile = '%s/ikdb%d.txt' % (robot.getName(),numTrain)
        mkdir_p(robot.getName())
    else:
        dbfile = trainfile
    if testfile == 'auto':
        testfile = '%s/iktest%d.txt' % (robot.getName(),numTest)
        mkdir_p(robot.getName())

    #instantiate the DB and features
    if multi:
        tester = MultiIKDBTester(robot)
        for (template,features,ranges) in zip(iktemplate,ikfeatures,featureranges):
            tester.db.getDatabase(template).setIKProblemSpace(template,features,ranges)
    else:
        tester = IKDBTester(robot)
        tester.db.setIKProblemSpace(iktemplate,ikfeatures,featureranges)

    if redoTrain or not os.path.exists(dbfile):
        #generating from scratch
        tester.generateDB(numTrain)
        tester.saveDB(dbfile)
    else:
        #load tests from testing file
        tester.loadDB(dbfile)

    if redoTest or not os.path.exists(testfile):
        #generating from scratch
        tester.generateTest(numTest)
        tester.saveTest(testfile)
    else:
        #load tests from testing file
        tester.loadTest(testfile)
    return tester

def default_position_range(link,localpoint,numIters=1000,expansionFactor=0.1):
    """Calculates a range of the position on the given robot link using random sampling
    + an expansion factor.
    """
    robot = link.robot()
    qmin,qmax = robot.getJointLimits()
    ranges = [None,None,None]
    for i,(a,b) in enumerate(zip(qmin,qmax)):
        if not np.isfinite(a):
            print("Warning, Setting finite bound on joint",i)
            qmin[i] = -math.pi
        if not np.isfinite(b):
            print("Warning, Setting finite bound on joint",i)
            qmax[i] = math.pi
        if not (b-a  < 1e20):
            print("Warning, robot has excessive motion range on link",robot.link(i).getName())
            input("Press enter to continue")
            break
    cnt = 0
    for i in range(10000):
        robot.setConfig([random.uniform(a,b) for (a,b) in zip(qmin,qmax)])
        if robot.selfCollides():
            continue

        wp = link.getWorldPosition(localpoint)
        if cnt == 0:
            for k in range(3):
                ranges[k] = (wp[k],wp[k])
        else:
            for k in range(3):
                ranges[k] = (min(ranges[k][0],wp[k]),max(ranges[k][1],wp[k]))
        cnt += 1
        if cnt > 1000:
            break
    print ("Randomized range on link",link.getName(),":",ranges)
    if cnt < 100:
        print ("WARNING: very few self-collision free configurations found")
    for k in range(3):
        exp = expansionFactor*(ranges[k][1]-ranges[k][0])
        ranges[k] = (ranges[k][0]-exp,ranges[k][1]+exp)
    return ranges

def run_tests_main(ikobjectivejson,ikfeatures):
    """A basic tester that runs a given ik feature as configured by the command line arguments.  Runs a given
    IK objective on one or more links"""
    import argparse
    import os
    default_fn = os.path.expanduser("~/Klampt-examples/data/robots/tx90ball.rob")
    default_parser = argparse.ArgumentParser(description='Generate an IK database for a point-constrained IK problem.')
    default_parser.add_argument('-t','--train',metavar='N',type=int,help="# of training examples",default=1000)
    default_parser.add_argument('-T','--test',metavar='N',type=int,help="# of testing examples",default=1000)
    default_parser.add_argument('-r','--robot',metavar='ROBOT',type=str,help="Robot or world file",default=default_fn)
    default_parser.add_argument('-f','--file',metavar='FILE',type=str,help="DB file name",default="auto")
    default_parser.add_argument('-g','--testfile',metavar='FILE',type=str,help="Test file name",default="auto")
    default_parser.add_argument('--redotrain',action='store_const',const=1,help="Re-train the database",default=0)
    default_parser.add_argument('--redotest',action='store_const',const=1,help="Re-generate the test set",default=0)
    default_parser.add_argument('--metriclearn',metavar='N',type=int,help="Number of iterations of metric learning",default=0)
    default_parser.add_argument('-k','--neighbors',metavar='K',action='append',type=int,help="Number of nearest neighbors for prediction",default=[1,5,10])
    default_parser.add_argument('--RR',metavar='K',action='append',type=int,help="Number of restarts to test",default=[])
    default_parser.add_argument('--cost',metavar='NAME',type=str,help="Name of objective function",default='jointRangeCost')
    default_parser.add_argument('--feasible',metavar='NAME',type=str,help="Name of feasibility test function",default='collisionFree')
    default_parser.add_argument('-p','--localpoint',metavar='V',action='append',type=float,nargs=3,help="Local point of IK constraint",default=[0,0,0])
    default_parser.add_argument('-l','--link',metavar='N',action='append',help="Index or name of robot link(s)",default=[])
    
    ns = default_parser.parse_args(sys.argv[1:])

    world = WorldModel()
    world.readFile(ns.robot)
    robot = world.robot(0)

    #setup the set of possible functions
    functionfactory.registerDefaultFunctions()
    functionfactory.registerCollisionFunction(world)
    functionfactory.registerJointRangeCostFunction(robot)

    #make the IK template, and its ranges and features
    objectives = []
    features = []
    ranges = []
    if len(ns.link)==0:
        print ("No link specified, using the last link in the robot file")
        ns.link = [robot.numLinks()-1]
    for i,link in enumerate(ns.link):
        try:
            link = int(link)
        except:
            pass

        #set up the IK problem templates
        link  = robot.link(link)
        ikobjectivejson['link'] = link.index
        ikobjectivejson['type'] = 'IKObjective'
        ikobjectivejson['localPosition'] = ns.localpoint
        obj = loader.fromJson(ikobjectivejson)
        obj.robot = robot
        objectives.append(obj)
        if isinstance(ikfeatures,(list,tuple)):
            for f in ikfeatures:
                features.append(('objectives',i,f))
        else:
            features.append(('objectives',i,ikfeatures))

        #auto-detect position range via random sampling
        linkrange = default_position_range(link,ns.localpoint)
        linkrange = linkrange[:obj.numPosDims()]
        linkrange += [(-math.pi,math.pi)]*obj.numRotDims()
        ranges += linkrange
    template = IKProblem(*objectives)
    
    #initialize cost function and feasibility test
    template.setFeasibilityTest(ns.feasible,None)
    if ns.cost == 'jointRangeCost':
        qmin,qmax = robot.getJointLimits()
        template.setCostFunction('jointRangeCost_dynamic',[qmin,qmax])
    else:
        template.setCostFunction(ns.cost,None)

    tester = make_ikdb_tester(robot,template,features,ranges,
        ns.train,ns.test,
        ns.file,ns.testfile,
        ns.redotrain,ns.redotest)

    if ns.metriclearn > 0:
        for i in range(10):
            tester.db.metricLearn(ns.metriclearn/10)
            print ("Round",i,"matrix:",tester.db.metricMatrix)

    #run the testing
    for N in ns.RR:
        tester.testRaw(N)
    for k in ns.neighbors:
        tester.testDB(k)

if __name__=='__main__':
    #example
    #you can create the template by 1. creating an ik objective, then using the loader.toJson function as follows:
    #pointiktemplate = loader.toJson(ik.objective(0,local=[0,0,0],world=[0,0,0]))
    #Or 2. creating a JSON structure directly, see Klampt's IKObjective JSON format 
    pointiktemplate = {'posConstraint':'fixed','localPosition':[0,0,0],'endPosition':[0,0,0]}
    pointikfeatures = 'endPosition'
    #this creates an xform objective
    #method 1
    #xformiktemplate = loader.toJson(ik.objective(0,R=so3.identity(),t=[0,0,0]))
    #method 2
    #xformiktemplate = {'posConstraint':'fixed','localPosition':[0,0,0],'endPosition':[0,0,0],'rotConstraint':'fixed','endRotation':[0,0,0]}
    #xformikfeatures = 'endPosition'
    run_tests_main(pointiktemplate,pointikfeatures)
