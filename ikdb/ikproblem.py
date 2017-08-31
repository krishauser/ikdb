import pkg_resources
if pkg_resources.get_distribution('klampt').version >= '0.7':
    NEW_KLAMPT = True
    from klampt.model import ik
    from klampt.math import vectorops
    from klampt import IKObjective
    from klampt.io import loader
else:
    NEW_KLAMPT = False
    from klampt import ik,vectorops
    from klampt import IKObjective
    from klampt import loader
import time
import features
import random
import optimize
import functionfactory

class IKSolverParams:
    def __init__(self,numIters=50,tol=1e-3,
                 startRandom=False,numRestarts=1,
                 timeout=10,globalMethod=None,localMethod=None):
        self.numIters=numIters
        self.tol=tol
        self.startRandom=startRandom
        self.numRestarts=numRestarts
        self.timeout = timeout
        self.globalMethod = globalMethod
        self.localMethod = localMethod

def global_solve(optproblem,params=IKSolverParams(),seed=None):
    """Globally solves a optimize.Problem instance with the given IKSolverParams.
    Optionally takes a seed as well.
    """
    method = params.globalMethod
    numIters = params.numIters
    tol = params.tol
    if params.globalMethod == 'random-restart':
        #use the GlobalOptimize version of random restarts
        assert params.localMethod is not None,"Need a localMethod for random-restart to work"
        method = params.globalMethod + '.' + params.localMethod
        numIters = [params.numRestarts,params.numIters]
        optSolver = optimize.GlobalOptimizer(method)
    elif params.localMethod is not None:
        #do a sequential optimization
        method = [params.globalMethod,params.localMethod]
        #co-opt params.numRestarts for the number of outer iterations?
        numIters = [params.numRestarts,params.numIters]
    optSolver = optimize.GlobalOptimizer(method=method)
    if seed:
        optSolver.setSeed(seed)
    (succ,res) = optSolver.solve(optProblem,numIters=numIters,tol=tol)
    return (succ,res)

class IKProblem:
    """Defines a generalized IK problem that can be saved/loaded from a JSON
    string.  It may have additional specifications of active DOF, feasibility
    tests, or cost functions.

    Note that feasibility tests and cost functions must be savable/loadable
    with a serializable description.  Specifically, the function
    functionfactory.makeFunction must be able to create it.  There are
    several default function types, listed above, which are made available
    via the calls:
    
    functionfactory.registerDefaultFunctions()
    functionfactory.registerFunction('funcName',function)
    #or to specify that the configuration variable has a name other than 'x'
    #in function's, specify a third argument.  For example, if you call it q,
    #then use this option:
    functionfactory.registerFunction('funcName',function,'q')

    If you would like to find the configuration *closest* to solving the IK
    constraints, call setSoftObjectives().  In this case, solve will always
    return a solution, as long as it finds one that passes the feasibility test.
    The optimization method changes so that it 1) optimizes the IK residual norm, and
    then 2) optimizes the cost function to maintain the residual norm at its
    current value.  In other words, minimizing IK error is the first priority and
    minimizing cost is the second priority.
    """
    def __init__(self,*ikgoals):
        self.objectives = list(ikgoals)
        self.softObjectives = False
        self.activeDofs = None
        self.jointLimits = None
        self.costFunction = None
        self.costFunctionDescription = None
        self.feasibilityTest = None
        self.feasibilityTestDescription = None
    def addObjective(self,obj):
        """Adds a new IKObjective to the problem."""
        assert isinstance(obj,IKObjective)
        self.objectives.append(obj)
    def addConstraint(self,obj):
        """An alias to addObjective. """
        self.addObjective(obj)
    def setSoftObjectives(self,enabled=True):
        """Turns on soft IK solving.  This is the same as hard IK solving if all
        constraints can be reached, but if the constraints cannot be reached, it will
        """
        self.softObjectives = enabled
    def setCostFunction(self,type,args=None):
        """Sets the current cost function to a function of type 'type' with
        parameters 'args'. See the documentaiton of the function registry to
        see how to build your own dynamically instantiable functions."""
        self.costFunction = functionfactory.makeFunction(type,args)
        if self.costFunction is None:
            raise ValueError("Invalid function type and/or args when setting cost function, must be registered with function registry")
        self.costFunctionDescription = {'type':type,'args':args}
    def setFeasibilityTest(self,type,args=None):
        """Sets the feasibility test to a function of type 'type' with parameters
        'args'.  See the documentation of the function registry to see how
        to build your own dynamically instantiable functions."""
        self.feasibilityTest = functionfactory.makeFunction(type,args)
        if self.feasibilityTest is None:
            raise ValueError("Invalid function type and/or args when setting feasibility test, must be registered with function registry")
        self.feasibilityTestDescription = {'type':type,'args':args}
    def addFeasibilityTest(self,type,args=None):
        """Adds an additional feasibility test.  See setFeasibilityTest for 
        documentation of these parameters."""
        if self.feasibilityTest is None:
            self.setFeasibilityTest(type,args)
            return
        else:
            f = functionfactory.makeFunction(type,args)
            self.feasibilityTest = functionfactory.andFunction(f,self.feasibilityTest)
            if isinstance(self.feasibilityTestDescription,list):
                self.feasibilityTestDescription.append({'type':type,'args':args})
            else:
                #first time
                self.feasibilityTestDescription = [self.feasibilityTestDescription,{'type':type,'args':args}]
    def setActiveDofs(self,links):
        """Sets the list of active DOFs."""
        self.activeDofs = links
    def enableDof(self,link):
        """Enables an active DOF.  If this is the first time enableDof is called,
        this initializes the list of active DOFs to the single link.  Otherwise
        it appends it to the list.  (By default, all DOFs are enabled)"""
        if self.activeDofs is None:
            self.activeDofs = [link]
        else:
            if link not in self.activeDofs:
                self.activeDofs.append(link)
    def disableJointLimits(self):
        """Disables joint limits.  By default, the robot's joint limits are
        used."""
        self.jointLimits = ([],[])
    def setJointLimits(self,qmin=None,qmax=None):
        """Sets the joint limits to the given lists qmin,qmax.  By default,
        the robot's joint limits are used."""
        if qmin is None:
            self.jointLimits = None
            return
        #error checking
        assert(len(qmin)==len(qmax))
        if len(qmin)==0:
            #disabled bounds
            self.jointLimits = (qmin,qmax)
        else:
            if self.activeDofs is not None:
                assert(len(qmin)==len(self.activeDofs))
            else:
                if len(self.objectives) != 0:
                    if hasattr(self.objectives[0],'robot'):
                        assert(len(qmin) == self.objectives[0].numLinks())
        self.jointLimits = (qmin,qmax)
    def toJson(self):
        """Returns a JSON object representing this IK problem."""
        res = dict()
        res['type'] = 'IKProblem'
        objectives = []
        for obj in self.objectives:
            objectives.append(loader.toJson(obj))
        res['objectives'] = objectives
        if self.softObjectives:
            res['softObjectives'] = self.softObjectives
        if self.activeDofs is not None:
            res['activeDofs'] = self.activeDofs
        if self.jointLimits is not None:
            res['jointLimits'] = self.jointLimits
        if self.costFunction is not None:
            res['costFunction'] = self.costFunctionDescription
        if self.feasibilityTest is not None:
            res['feasibilityTest'] = self.feasibilityTestDescription
        return res
    def fromJson(self,object):
        """Sets this IK problem to a JSON object representing it. A ValueError
        is raised if it is not the correct type."""
        if object['type'] != 'IKProblem':
            raise ValueError("Object must have type IKProblem")
        self.objectives = []
        for obj in object['objectives']:
            self.objectives.append(loader.fromJson(obj))
        if 'softObjectives' in object:
            self.softObjectives = object['softObjectives']
        else:
            self.softObjectives = False
        self.activeDofs = object.get('activeDofs',None)
        self.jointLimits = object.get('jointLimits',None)
        self.costFunctionDescription = object.get('costFunction',None)
        if self.costFunctionDescription is None:
            self.costFunction = None
        else:
            self.costFunction = functionfactory.makeFunction(self.costFunctionDescription['type'],self.costFunctionDescription['args'])
        self.feasibilityTestDescription = object.get('feasibilityTest',None)
        if self.feasibilityTestDescription is None:
            self.feasibilityTest = None
        else:
            if isinstance(self.feasibilityTestDescription,dict):
                self.feasibilityTest = functionfactory.makeFunction(self.feasibilityTestDescription['type'],self.feasibilityTestDescription['args'])
            else:
                self.feasibilityTest = functionfactory.andFunction(*[functionfactory.makeFunction(test['type'],test['args']) for test in self.feasibilityTestDescription]) 
        return
    def solve(self,robot=None,params=IKSolverParams()):
        """Globally solves the given problem.  Returns the solution
        configuration or None if failed."""
        #set this to False if you want to run the local optimizer for each
        #random restart.
        postOptimize = True
        t0 = time.time()
        if len(self.objectives) == 0:
            if self.costFunction is not None or self.feasibilityTest is not None:
                raise NotImplementedError("Raw optimization without IK goals not done yet")
            return None
        if robot is None:
            if not hasattr(self.objectives[0],'robot'):
                print "The objectives passed to IKSolver should come from ik.objective() or have their 'robot' member manually set"
            robot = self.objectives[0].robot
        else:
            for obj in self.objectives:
                obj.robot = robot
        solver = ik.solver(self.objectives)
        if self.activeDofs is not None:
            solver.setActiveDofs(self.activeDofs)
            ikActiveDofs = self.activeDofs
        if self.jointLimits is not None: solver.setJointLimits(*self.jointLimits)
        qmin,qmax = solver.getJointLimits()
        if self.activeDofs is None:
            #need to distinguish between dofs that affect feasibility vs 
            ikActiveDofs = solver.getActiveDofs()
            if self.costFunction is not None or self.feasibilityTest is not None:
                activeDofs = [i for i in range(len(qmin)) if qmin[i] != qmax[i]]
                nonIKDofs = [i for i in activeDofs if i not in ikActiveDofs]
                ikToActive = [activeDofs.index(i) for i in ikActiveDofs]
            else:
                activeDofs = ikActiveDofs
                nonIKDofs = []
                ikToActive = range(len(activeDofs))
        else:
            activeDofs = ikActiveDofs
            nonIKDofs = []
            ikToActive = range(len(ikActiveDofs))
        #sample random start point
        if params.startRandom:
            solver.sampleInitial()
            if len(nonIKDofs)>0:
                q = robot.getConfig()
                for i in nonIKDofs:
                    q[i] = random.uniform(qmin[i],qmax[i])
                robot.setConfig(q)
        if params.localMethod is not None or params.globalMethod is not None:
            #set up optProblem, an instance of optimize.Problem
            optProblem = optimize.Problem()
            Jactive = [[0.0]*len(activeDofs)]*len(solver.getResidual())
            def ikConstraint(x):
                q = robot.getConfig()
                for d,v in zip(activeDofs,x):
                    q[d] = v
                robot.setConfig(q)
                return solver.getResidual()
            if NEW_KLAMPT:
                def ikConstraintJac(x):
                    q = robot.getConfig()
                    for d,v in zip(activeDofs,x):
                        q[d] = v
                    robot.setConfig(q)
                    return solver.getJacobian()
            else:
                #old version of Klamp't didn't compute the jacobian w.r.t. the active DOFS
                def ikConstraintJac(x):
                    q = robot.getConfig()
                    for d,v in zip(activeDofs,x):
                        q[d] = v
                    robot.setConfig(q)
                    Jikdofs = solver.getJacobian()
                    for i in ikActiveDofs:
                        for j in xrange(len(Jactive)):
                            Jactive[j][ikToActive[i]] = Jikdofs[j][i]
                    return Jactive
            def costFunc(x):
                q = robot.getConfig()
                for d,v in zip(activeDofs,x):
                    q[d] = v
                return self.costFunction(q)
            def feasFunc(x):
                q = robot.getConfig()
                for d,v in zip(activeDofs,x):
                    q[d] = v
                return self.feasibilityTest(q)
            optProblem.addEquality(ikConstraint,ikConstraintJac)
            if len(qmax) > 0:
                optProblem.setBounds([qmin[d] for d in activeDofs],[qmax[d] for d in activeDofs])
            if self.costFunction is None:
                optProblem.setObjective(lambda x:0)
            else:
                optProblem.setObjective(costFunc)
            if self.feasibilityTest is not None:
                optProblem.setFeasibilityTest(feasFunc)
            #optProblem is now ready to use

            if self.softObjectives:
                softOptProblem = optimize.Problem()
                def costFunc(x):
                    q = robot.getConfig()
                    for d,v in zip(activeDofs,x):
                        q[d] = v
                    robot.setConfig(q)
                    return vectorops.normSquared(solver.getResidual())*0.5
                def costFuncGrad(x):
                    q = robot.getConfig()
                    for d,v in zip(activeDofs,x):
                        q[d] = v
                    robot.setConfig(q)
                    return solver.getResidual()
                if len(qmax) > 0:
                    softOptProblem.setBounds([qmin[d] for d in activeDofs],[qmax[d] for d in activeDofs])
                if self.feasibilityTest is not None:
                    softOptProblem.setFeasibilityTest(feasFunc)
                softOptProblem.setObjective(costFunc,costFuncGrad)
                #softOptProblem is now ready to use

        if params.globalMethod is not None:
            q = robot.getConfig()
            x0 = [q[d] for d in activeDofs]
            if self.softObjectives:
                #globally optimize the soft objective function.  If 0 objective value is obtained, use equality constrained
                #optProblem.  If 0 objective value is not obtained, constrain the residual norm-squared to be that value
                (succ,res) = global_solve(softOptProblem,params,x0)
                if not succ:
                    print "Global soft optimize returned failure"
                    return None
                for d,v in zip(activeDofs,res):
                    q[d] = v
                if self.costFunction is None:
                    #no cost function, just return
                    print "Global optimize succeeded! Cost",self.costFunction(q)
                    return q
                x0 = res
                #now modify the constraint of optProblem
                robot.setConfig(q)
                residual = solver.getResidual()
                if max(abs(v) for v in residual) < params.tol:
                    #the constraint is satisfied, now just optimize cost
                    pass
                else:
                    #the constraint is not satisfied, now use the residual as the constraint
                    threshold = 0.5*vectorops.normSquared(residual)
                    def inequality(x):
                        q = robot.getConfig()
                        for d,v in zip(activeDofs,x):
                            q[d] = v
                        robot.setConfig(q)
                        return [vectorops.normSquared(solver.getResidual())*0.5 - threshold]
                    def inequalityGrad(x):
                        return [costFuncGrad(x)]
                    optProblem.equalities = []
                    optProblem.equalityGrads = []
                    optProblem.addInequality(inequality,inequalityGrad)

            #do global optimization of the cost function and return
            (succ,res) = global_solve(optProblem,params,x0)
            if not succ:
                print "Global optimize returned failure"
                return None
            for d,v in zip(activeDofs,res):
                q[d] = v
            #check feasibility if desired
            if self.feasibilityTest is not None and not self.feasibilityTest(q):
                print "Result from global optimize isn't feasible"
                return None
            if not softObjectives:
                if max(abs(v) for v in solver.getResidual()) > params.tol:
                    print "Result from global optimize doesn't satisfy tolerance.  Residual",vectorops.norm(solver.getResidual())
                    return None
            #passed
            print "Global optimize succeeded! Cost",self.costFunction(q)
            return q                

        #DONT DO THIS... much faster to do newton solves first, then local optimize.
        if not postOptimize and params.localMethod is not None:
            if self.softObjectives:
                raise RuntimeError("Direct local optimization of soft objectives is not done yet")
            #random restart + local optimize
            optSolver = optimize.LocalOptimizer(method=params.localMethod)
            q = robot.getConfig()
            x0 = [q[d] for d in activeDofs]
            optSolver.setSeed(x0)
            best = None
            bestQuality = float('inf')
            for restart in xrange(params.numRestarts):
                if time.time() - t0 > params.timeout:
                    return best
                res = optSolver.solve(optProblem,params.numIters,params.tol)
                if res[0]:
                    q = robot.getConfig()
                    for d,v in zip(activeDofs,res[1]):
                        q[d] = v
                    #check feasibility if desired
                    if self.feasibilityTest is not None and not self.feasibilityTest(q):
                        continue
                    if self.costFunction is None:
                        #feasible
                        return q
                    else:
                        #optimize
                        quality = self.costFunction(q)
                        if quality < bestQuality:
                            best = q
                            bestQuality = quality
                #random restart
                solver.sampleInitial()
                q = robot.getConfig()
                if len(nonIKDofs)>0:
                    for i in nonIKDofs:
                        q[i] = random.uniform(qmin[i],qmax[i])
                x0 = [q[d] for d in activeDofs]
                optSolver.setSeed(x0)
        else:
            #random-restart newton-raphson
            solver.setMaxIters(params.numIters)
            solver.setTolerance(params.tol)
            best = None
            bestQuality = float('inf')
            if self.softObjectives:
                #quality is a tuple
                bestQuality = bestQuality,bestQuality
            quality = None
            for restart in xrange(params.numRestarts):
                if time.time() - t0 > params.timeout:
                    return best
                t0 = time.time()
                res = solver.solve()
                if res or self.softObjectives:
                    q = robot.getConfig()
                    #check feasibility if desired
                    t0 = time.time()
                    if self.feasibilityTest is not None and not self.feasibilityTest(q):
                        if len(nonIKDofs) > 0:
                            u = float(restart+0.5)/params.numRestarts
                            q = robot.getConfig()
                            #perturbation sampling
                            for i in nonIKDofs:
                                delta = u*(qmax[i]-qmin[i])*0.5
                                q[i] = random.uniform(max(q[i]-delta,qmin[i]),min(q[i]+delta,qmax[i]))
                            robot.setConfig(q)
                            if not self.feasibilityTest(q):
                                solver.sampleInitial()
                                continue
                        else:
                            solver.sampleInitial()
                            continue
                    if self.softObjectives:
                        residual = solver.getResidual()
                        ikerr = max(abs(v) for v in residual)
                        if ikerr < params.tol:
                            ikerr = 0
                        else:
                            #minimize squared error
                            ikerr = vectorops.normSquared(residual)
                        if self.costFunction is None:
                            cost = 0
                            if ikerr == 0:
                                #feasible and no cost
                                return q
                        else:
                            cost = self.costFunction(q)
                        quality = ikerr,cost
                    else:
                        if self.costFunction is None:
                            #feasible
                            return q
                        else:
                            #optimize
                            quality = self.costFunction(q)
                    if quality < bestQuality:
                        best = q
                        bestQuality = quality
                #sample a new ik seed
                solver.sampleInitial()

        #post-optimize using local optimizer
        if postOptimize and best is not None and params.localMethod is not None:
            if self.softObjectives:
                robot.setConfig(best)
                residual = solver.getResidual()
                if max(abs(v) for v in residual) > params.tol:
                    #the constraint is not satisfied, now use the residual as the constraint
                    threshold = 0.5*vectorops.normSquared(residual)
                    def inequality(x):
                        q = robot.getConfig()
                        for d,v in zip(activeDofs,x):
                            q[d] = v
                        robot.setConfig(q)
                        return [vectorops.normSquared(solver.getResidual())*0.5 - threshold]
                    def inequalityGrad(x):
                        return [costFuncGrad(x)]
                    optProblem.equalities = []
                    optProblem.equalityGrads = []
                    optProblem.addInequality(inequality,inequalityGrad)
            optSolver = optimize.LocalOptimizer(method=params.localMethod)
            x0 = [best[d] for d in activeDofs]
            optSolver.setSeed(x0)
            res = optSolver.solve(optProblem,params.numIters,params.tol)
            if res[0]:
                q = robot.getConfig()
                for d,v in zip(activeDofs,res[1]):
                    q[d] = v
                #check feasibility if desired
                if self.feasibilityTest is not None and not self.feasibilityTest(q):
                    pass
                elif self.costFunction is None:
                    #feasible
                    best = q
                else:
                    #optimize
                    quality = self.costFunction(q)
                    if quality < bestQuality:
                        #print "Optimization improvement",bestQuality,"->",quality
                        best = q
                        bestQuality = quality
                    elif quality > bestQuality + 1e-2:
                        print "Got worse solution by local optimizing?",bestQuality,"->",quality
        return best
    
    def score(self,robot):
        """Returns an error score that is equal to the optimum at a feasible
        solution. Evaluated at the robot's current configuration."""
        for obj in self.objectives:
            obj.robot = robot
        solver = ik.solver(self.objectives)
        c = (0 if self.costFunction is None else self.costFunction(robot.getConfig()))
        return c+vectorops.norm(solver.getResidual())

    def constraintResidual(self,robot):
        """Returns a residual of the constraints at the robot's configuration."""
        for obj in self.objectives:
            obj.robot = robot
        solver = ik.solver(self.objectives)
        return solver.getResidual()


def ikObjectiveToFeatures(ikgoal,featureList):
    """Given an IKObjective instance and a list of features, returns a
    feature descriptor as a list of floats"""
    jsonObj = loader.toJson(ikgoal)
    return features.extract(jsonObj,featureList)

def ikProblemToFeatures(ikproblem,featureList):
    """Standard feature extractor for IKProblems"""
    if isinstance(ikproblem,dict):
        #assume it's a JSON object already
        return features.extract(ikproblem,featureList)
    elif isinstance(ikproblem,IKProblem):
        jsonObj = ikproblem.toJson()
        return features.extract(jsonObj,featureList)
    elif isinstance(ikproblem,IKObjective):
        return ikObjectiveToFeatures(ikproblem,featureList)
    else:
        assert hasattr(ikproblem,'__iter__'),"IK problem must either be an IKProblem, single IKObjective, or a list"
        return sum([ikObjectiveToFeatures(o,f) for o,f in zip(ikproblem,featureList)],[])

def featuresToIkObjective(ikgoal0,featureList,values):
    """Given an IKObjective "template", a list of features, and a list of values,
    returns a new IKObjective whose features are set to the list of values"""
    jsonObj = loader.toJson(ikgoal0)
    features.inject(jsonObj,featureList,values)
    obj = loader.fromJson(jsonObj,type='IKObjective')
    if hasattr(ikgoal0,'robot'):
        obj.robot = ikgoal0.robot
    return obj

def featuresToIkProblem(ikproblem0,featureList,values):
    """Standard feature injector for IKProblems"""
    if isinstance(ikproblem0,dict):
        #assume it's a JSON object already
        features.inject(ikproblem0,featureList,values)
        obj = IKProblem()
        obj.fromJson(ikproblem0)
        return obj
    else:
        print "featureToIkProblem: slower version being called"
        raw_input()
    if isinstance(ikproblem0,IKProblem):
        jsonObj = ikproblem0.toJson()
        features.inject(jsonObj,featureList,values)
        obj = IKProblem()
        obj.fromJson(jsonObj)
        obj.objectives[0].robot = ikproblem0.objectives[0].robot
        return obj
    if isinstance(ikproblem0,IKObjective):
        return featuresToIkObjective(ikproblem0,featureList,values)
    else:
        assert hasattr(ikproblem0,'__iter__'),"Template IK problem must either be a single IKObjective or a list"
        viter = iter(values)
        return sum([featuresToIkObjective(o,f,viter) for o,f in zip(ikproblem0,featureList)],[])
