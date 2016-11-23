from klampt import *
from ikdb import ikdb
from ikdb import functionfactory
from klampt import loader
import sys
import math
import random
import time

#preload to avoid later delays
from sklearn.neighbors import NearestNeighbors,BallTree
from scipy.optimize import differential_evolution

global ns,robot
robot = None
ns = None

def init_main():
    global ns,robot
    import argparse
    default_parser = argparse.ArgumentParser(description='Generate an IK database for a point-constrained IK problem.')
    default_parser.add_argument('-r','--robot',metavar='ROBOT',type=str,help="Robot or world file",default="../../Klampt/data/robots/tx90ball.rob")
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

def run_tests_main(ikobjectivejson,ikfeatures):
    """A basic tester that runs a given ik feature as configured by the command line arguments.  Runs a given
    IK objective on one or more links"""

    #make the IK template from the json spec
    objectives = []
    if len(ns.link)==0:
        print "No link specified, using the last link in the robot file"
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

    #generate some IK problem
    res = ikdb.solve(objectives,feasibilityCheck=ns.feasible,costFunction=ns.cost)
    print "res:",res
    

if __name__=='__main__':
    init_main()
    #example
    #you can create the template by 1. creating an ik objective, then using the loader.toJson function as follows:
    #pointiktemplate = loader.toJson(ik.objective(0,local=[0,0,0],world=[0,0,0]))
    #Or 2. creating a JSON structure directly, see Klampt's IKObjective JSON format 
    pointiktemplate = {'posConstraint':'fixed','localPosition':[0,0,0],'endPosition':[1,0,0.5]}
    pointiktemplate2 = {'posConstraint':'fixed','localPosition':[0,0,0],'endPosition':[0,1,0.6]}
    pointiktemplate3 = {'posConstraint':'fixed','localPosition':[0,0,0],'endPosition':[0.5,0.5,0.3]}
    pointikfeatures = 'endPosition'
    #this creates an xform objective
    #method 1
    #xformiktemplate = loader.toJson(ik.objective(0,R=so3.identity(),t=[0,0,0]))
    #method 2
    #xformiktemplate = {'posConstraint':'fixed','localPosition':[0,0,0],'endPosition':[0,0,0],'rotConstraint':'fixed','endRotation':[0,0,0]}
    #xformikfeatures = 'endPosition'

    #these will seed the IKDB with some solve queries
    run_tests_main(pointiktemplate,pointikfeatures)
    run_tests_main(pointiktemplate2,pointikfeatures)
    run_tests_main(pointiktemplate3,pointikfeatures)

    #wait for a minute before quitting to allow the DB to generate examples int he background
    time.sleep(60)
    #this saves the database to disk
    ikdb.flush()

