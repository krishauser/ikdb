# Inverse Kinematics Database (IKDB) Library #
### version 0.3 ###

Kris Hauser

University of Illinois at Urbana Champaign

kkhauser@illinois.edu

1/15/2020

## 1. Purpose ##

The IKDB package is a learning-based approach for building large databases of 
global, collision-free, optimal solutions of redundant IK problems.  It addresses
the problem that existing numerical solutions for IK use local optimization, which 
tends to fall into local minima due to joint limits, and they do not properly take
collision avoidance into account.  The approach taken by IKDB is to pre-train a 
(usually large) database of globally-optimized solutions offline, and then adapt them
online using local optimization.  With a properly trained database, the resulting IK
solver is orders of magnitude faster than standard global optimization techniques.

It also provides functionality to learn the database in the background.

It accompanies the paper:

    Kris Hauser, Learning the Problem-Optimum Map: Analysis and Application
    to Global Optimization in Robotics. IEEE Transactions on Robotics, 2017. [also appearing in
    arXiv:1605.04636, http://arxiv.org/abs/1605.04636]

IKDB is written in a Python front end for customizability, while the solvers used
by its dependencies use C++ and Fortran back ends for speed.


## 2. Installation and dependencies ##

IKDB requires Python and the following Python packages:
- Scipy
- Scikit-learn
- Klampt Python API (http://klampt.org) version 0.6.X, 0.7.X, and 0.8.X supported
- Optional packages:
  - PyOpt (http://www.pyopt.com), a local optimization package 
  - DIRECT (https://pypi.python.org/pypi/DIRECT/), another global optimizer used
    for comparison.  (Testing indicates performance is not competitive.)
  - PyOpenGL, for visualization in ikdbtest_gl.py

Simply use `pip install klampt scipy sklearn PyOpenGL` to install most of these packages.
PyOpt and DIRECT should be installed from scratch.

For more Klamp't install options, follow the installation tutorial for your system at
[http://motion.cs.illinois.edu/klampt/tutorial_install.html](http://motion.cs.illinois.edu/klampt/tutorial_install.html)

## 3. Concepts ##

### A. Overall problem ###

IKDB is meant to address global optimization problems of the form:

>     minimize over q in R^n the cost function f(q)
>            such that
>        qmin <= q <= qmax
>      E_ik(q,constraint1) = 0
>      E_ik(q,constraint2) = 0                               (1)
>               ...
>      E_ik(q,constraintr) = 0
>       q lies in the set F

where E_ik(q,constraint) gives the constraint error function of an inverse kinematics
constraint, and F is a feasibility tester (e.g., testing that a configuration is
collision free).

These are fairly time consuming to solve globally in real-time for complex robots.
Global optimization techniques take on the order of seconds to solve them
approximately.

The idea behind IKDB is to store several related, parameterized optimization problems
in the same form as (1).  Specifically, we define what is known as a P-parameter

>    z in R^m

that may affect the cost function and constraints of (1).  In other words, we have:

>     f(q) == f(q,z)
>     constraint1 == constraint1(z) 
>     ...
>     constraintr == constraintr(z)

(and in general, qmin, qmax, and F may also vary as a function of z as well).

IKDB solves a large number of similar problems P-parameter space z1,...,zN to give
optimal or near-optimal solutions q1,...,qN, with the approximation qi ~= q*(zi). 
The database is formed of the pairs D={(q1,z1),...,(qN,zN)}.

Then, for a query problem with novel P-parameter z, IKDB looks up the k-closest problems
zi1,...,zik in D, and tries to solve the problem defined by z by seeding a local optimizer
with the previous solutions qi1,...,qik until one works.  This local optimization is
usually very fast.

### Quick start ###

We have three ways of interacting with the IKDB module, listed in increasing
order of complexity and power: 
- simplified Klampt ik module API replacement interface
- automatically managed database
- manually managed databases

The easiest way to start is to use the ikdb module as an almost drop-in
replacement for the Klamp't ik module.  All of the functions of the ik API (objective,
solve, solve_nearby, etc) are duplicated in ikdb. Code for running a background running,
automatically learning IKDB is as follows:

```python
import ikdb
import klampt
import time

world = klampt.WorldModel()
#TODO: load the robot or world file that you will be using
world.loadFile(###[URDF, Klamp't .rob, or Klamp't .xml file]###)
ikdb.functionfactory.registerDefaultFunctions()
ikdb.functionfactory.registerCollisionFunction(world)
ikdb.functionfactory.registerJointRangeCostFunction(world.robot(0))

robot = world.robot(0)
while (True):
  #generate a new problem
  #TODO: add Klamp't IKObjective objects to the list of objectives
  problem = []
  #problem.append(ik.objective(robot.link(...),...))
  #problem.append(ik.objective(robot.link(...),...))
  #...
  
  #run the solver
  soln = ikdb.solve(problem,activeDofs=None,feasibilityCheck='collisionFree',costFunction='jointRangeCost')
  if soln is not None:
      print("Got a solution",soln)

#if you want to auto-populate the database, run the following lines:
time.sleep(600)
ikdb.flush()
```

Please consult the [Klamp't IK manual](http://motion.cs.illinois.edu/klampt/pyklampt_docs/Manual-IK.html) or
the [klampt.ik module documentation](http://motion.cs.illinois.edu/klampt/pyklampt_docs/klampt.model.ik.html)
for more information about how to set up these objectives.

The second easiest way to start is to use the ManagedIKDatabase class.  Here
you get to configure the folder in which the solver saves its database, control
the background loop, etc. Code is as follows:

```python
import ikdb
import klampt
import time

world = klampt.WorldModel()
#TODO: load the robot or world file that you will be using
world.loadFile(###[URDF, Klamp't .rob, or Klamp't .xml file]###)
ikdb.functionfactory.registerDefaultFunctions()
ikdb.functionfactory.registerCollisionFunction(world)
ikdb.functionfactory.registerJointRangeCostFunction(world.robot(0))

#create the IKDB, with optional folder to save in
db = ikdb.ManagedIKDatabase(world.robot(0),###[folder]###)
#optional: start the background thread.  It will automatically learn
#a database as it runs.  If you don't start it, then IKDB will learn
#only from the examples that you give it.
db.startBackgroundLoop()
while (True):
	#generate a new problem
    problem = ikdb.IKProblem()
    #TODO: make a list of Klamp't IKObjective objects, called objectives
    for obj in objectives:
    	problem.addConstraint(objective)
    #these ensure collision freeness and penalize proximity to joint limits
    problem.setFeasibilityTest('collisionFree')
    problem.setCostFunction('jointRangeCost')

    #run the solver
    soln = db.solve(problem)
    if soln is not None:
        print("Got a solution",soln)
#this is not strictly necessary, but you may want to do it just to be nice...
db.stopBackgroundLoop()
```

The ManagedIKDatabase will keep learning and periodically saving to disk, and the
learned database will be loaded up again next time you start the program.

The ManagedIKDatabase class automatically populates one or more IKDatabase objects and automatically
determines feature spaces from the IKProblems that you generate.  Doing this dynamically does 
use a little bit of overhead.  Instead, if you are solving a group of IKProblems that have
the same characteristics, it is a bit faster to call
```python
  sub_db = db.getDatabase(problem[0])
  db.solveWithDatabase(problem[0],sub_db)
  ...
  db.solveWithDatabase(problem[N],sub_db)
```
Note that this needs to be done AFTER database training.

You can also generate and evaluate
individual IKDatabase objects for fixed classes of IKProblems, but this requires a bit more
work, as described below.

### Robots and environment geometries ###

IKDB uses Klampt's data structures to represent robots and worlds.  In both training and
testing phases, users of IKDB will have to provide the same Klamp't RobotModel and
WorldModel instances to the solver.  Robots can be loaded in URDF format, and environment
geometries can be loaded in a wide variety of CAD formats, as well as the Point Cloud Data
(PCD) format.

Consult the examples or Klampt's documentation to learn more about how to load robot and world
files.

### Cost function and feasibility test customization ###

In order to customize the feasibility test and cost function, your functions need to be
serializable and instantiated dynamically.  The ikdb.functionfactory module is used for
this functionality.  Perhaps the easiest way to learn how to use it is through an example.

Consider a simple cost function penalizing the third joint's deviation from 0.5:

```python
  def my_simple_cost_fn(q):
	  return (q[2] - 0.5) ** 2
```

To register this, call:

```python
   functionfactory.registerFunction('my_cost',my_simple_cost_fn,'q')
```

Now it is ready to use in your IKProblem by calling

```python
  problem.setCostFunction('my_cost',None)
```

where the second argument tells the problem that your function takes no arguments
except for q.  This is assumed by default, so you can eliminate the None argument.

For functions that take arguments, like penalization from some start configuration, you
can easily add them to the definition, and IKDB will take care of them automatically.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def difference_cost_function(q,qref):
	  return sum((a - b) ** 2 for (a,b) in q,qref)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To register this, call:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   functionfactory.registerFunction('diff',difference_cost_function,'q')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, you can dynamically define a reference configuration in your IKProblem as follows

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  some_qref = [...] #set some reference configuration
  problem.setCostFunction('diff',some_qref)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If some_qref varies, then it will be determined to be part of the problem feature space.

A number of existing functions are provided for you, including
- 'linear' takes a float c0 and array-like c1
- 'quadratic' takes a float c0, array-like c1, and matrix-like c2
- 'distance_L1' computes L1 distance from a reference position
- 'distance_Linf' computes L-infinity distance from a reference position
- 'distance_L2' computes L2 distance from a reference position
- 'distance_squared_L2' computes sqared L2 distance from a reference position

### IKProblem ###

The key unit is the IKProblem, which defines an optimal, collision free IK problem.
This class is defined in the ikdb.ikproblem module, and stores the list of IK constraints,
an optional feasibility test, and an optional cost function to be minimized.  

Each IK constraint is defined as a Klamp't IKObjective object. 
Please consult the [Klamp't IK manual](http://motion.cs.illinois.edu/klampt/pyklampt_docs/Manual-IK.html) or
the [klampt.ik module documentation](http://motion.cs.illinois.edu/klampt/pyklampt_docs/klampt.model.ik.html)
for more information about how to set up these objectives.

IKProblem must be a JSON serializable class using the methods toJson and fromJson.  As a result
any arguments to custom-defined functions must have a JSON serialization.  In practice this
means you can only use primitive data types, lists and tuples, and dicts as arguments.

### Feature mappings ###

The way in which P-parameters z maps to changes of the functions in the problem is known as the
feature mapping.  Specifically, the feature mapping is a list of JSON paths, each of which references
floating-point elements in IKProblem JSON structures.

ManagedIKDatabase will learn these for you by detecting changes in IKProblem JSON structures.
But if you wish to define IKDatabases manually, or learn more about how this is done under
the hood, read on.

A simple problem with two IK objectives and no cost function will be defined in JSON format
like this:
```python
{
  type:"IKProblem",
  objectives:[
    {
   	  type:"IKObjective",
   	  link:6,
   	  posConstraint:"fixed",
   	  localPosition:[0,0,0],
   	  endPosition:[3.4,0.14,1.43]
   	},
    {
      type:"IKObjective",
      link:13,
      posConstraint:"fixed",
      localPosition:[0,0,0],
      endPosition:[-3.4,0.14,1.43]
      rotConstraint:"fixed",
      endRotation:[0,0,0]
    }
  ]
}
```

Each feature is a path through this hierarchical data structure.  That is, it is a list
of indices that will be traversed down the hierarchy.  As an example, if you wish toindicate
that the end position of only the x and y coordinates of the first constraint's world position
should be considered variable features, you would use the feature mapping:

>   [('objectives',0,'endPosition',0), ('objectives',0,'endPosition',1) ]

If all the x-y-z coordinates should be used, you can use the mapping

>   [('objectives',0,'endPosition',0), ('objectives',0,'endPosition',1), ('objectives',0,'endPosition',2)]

or the shortcut

>   [('objectives',0,'endPosition')]

In this latter case, the feature mapper will be smart about treating the 3-list as 3 separate features.

To treat both endpoints as features you can use the mapping:

>   [('objectives',0,'endPosition'),('objectives',1,'endPosition')]

and the feature mapping will be smart and provide a length 6 feature vector for this problem.

## 4. Test programs ##

* ikdbtest_simple.py: shows the use of the simplified API that duplicates the Klamp't ik module
  API.
* ikdbtest_console.py: conducts training and performance testing of the method
* ikdbtest_gl.py: trains a database from dynamically-defined IK problems in a visualization
  GUI (requires PyOpenGL) 

**Examples:**

These assume that [Klampt-examples](https://github.com/krishauser/Klampt-examples) is downloaded to
your home directory.  Try `cd ~; git clone https://github.com/krishauser/Klampt-examples`.

(basic test)

>   python ikdbtest_console.py --train 100000 --test 1000 --robot ~/Klampt-examples/data/robots/tx90ball.rob 

(tests against random-restart)

>   python ikdbtest_console.py --train 100000 --test 1000 --robot ~/Klampt-examples/data/robots/tx90ball.rob -k 1 -k 5 -k 10 --RR 1 --RR 10 --RR 100

(two links constrained)

>   python ikdbtest_console.py --train 100000 --test 1000 --link left_gripper --link right_gripper --robot ~/Klampt-examples/data/robots/baxter_col.rob 

(visualization and background training)

>   python ikdbtest_gl.py ~/Klampt-examples/data/robots/baxter_col.rob 

(simplified interface and background training)

>   python ikdbtest_simple.py ~/Klampt-examples/data/robots/baxter_col.rob 

## 5. Version history ##

- 0.1 (5/16/2016) - initial release
- 0.2 (11/24/2016) - revision for TRO release.  Added more documentation, ability to handle soft constraints, and Klampt 0.7 support.
- 0.3 (1/15/2020) - Updated to be compatible with Python 2/3. Works with Klampt 0.8.x
