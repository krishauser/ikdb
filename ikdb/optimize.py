class Problem:
    """A holder for optimization problem data.  All attributes are optional,
    and some solvers can't handle certain types of problem data.
    
    - objective: an objective function f(x)
    - objectiveGrad: a function df/dx(x) giving the gradient of f.
    - bounds: a pair (l,u) giving lower and upper bounds on the search space.
    - equalities: a list of functions g(x)=0 required of a feasible solution.
      In practice, |g(x)| <= tol is required.
    - equalityGrads: a list of gradient/Jacobian functions dg/dx(x) of the 
      equality functions.
    - inequalities: a list of functions h(x)<=0 required of a feasible
      solution.
    - inequalityGrads: a list of gradient/Jacobian functions dh/dx(x) of the 
      inequality functions.
    """
    def __init__(self):
        self.objective = None
        self.objectiveGrad = None
        self.bounds = None
        self.equalities = []
        self.inequalities = []
        self.equalityGrads = []
        self.inequalityGrads = []
        self.feasibilityTest = None
    def setObjective(self,func,funcGrad=None):
        self.objective = func
        self.objectiveGrad = funcGrad
    def addEquality(self,func,funcGrad=None):
        self.equalities.append(func)
        self.equalityGrads.append(funcGrad)
    def addInequality(self,func,funcGrad=None):
        self.inequalities.append(func)
        self.inequalityGrads.append(funcGrad)
    def setBounds(self,xmin,xmax):
        self.bounds = (xmin,xmax)
    def setFeasibilityTest(self,test):
        self.feasibilityTest = test
    def flatten(self,objective_scale,keep_bounds=True):
        """If this problem is constrained, returns a new problem in which
        the objective function is a scoring function that sums all of
        the equality / inequality errors at x plus
        objective_scale*objective function(x).  If objective_scale is small,
        then the scoring function is approximately minimized at a feasible
        minimum.

        If the problem is unconstrained, this just returns self.

        If keep_bounds = true, this does not add the bounds to the
        inequality errors.
        """
        #create a scoring function that is approximately minimized at
        #a feasible minimum
        if keep_bounds == False:
            raise NotImplementedError("getting rid of bounds is not implemented yet")
        if self.feasibilityTest is None and len(self.inequalities) == 0 and len(self.equalities) == 0:
            return self;

        if len(self.inequalities) == 0 and len(self.equalities) == 0:
            #just have a feasibility test
            def flatObjective(x):
                if not self.feasibilityTest(x):
                    return float('inf')
                return self.objective(x)
            res = Problem()
            res.setObjective(flatObjective,self.objectiveGrad)
            res.bounds = self.bounds
            return res

        def flatObjective(x):
            if self.feasibilityTest is not None:
                if not self.feasibilityTest(x):
                    return float('inf')
            f = 0
            #add sum of squared equalities
            for g in self.equalities:
                gx = g(x)
                f += max(abs(v) for v in gx)
            for h in self.inequalities:
                hx = h(x)
                f += sum(max(v,0) for v in hx)
            if self.objective is not None:
                f += objective_scale*self.objective(x)
            return f
        
        res = Problem()
        res.setObjective(flatObjective,None)
        res.bounds = self.bounds
        return res

class LocalOptimizer:
    """A wrapper around different local optimization libraries. Only
    minimization is supported, and only scipy and pyOpt are supported.
    
    The method is specified using the method string, which can be:
    - auto: picks between scipy and pyOpt, whatever is available.
    - scipy: uses scipy.optimize.minimize with default settings.
    - scipy.[METHOD]: uses scipy.optimize.minimize with the argument
      method=[METHOD].
    - pyOpt: uses pyOpt with SLSQP.
    - pyOpt.[METHOD]: uses pyOpt with the given method.
    """
    def __init__(self,method='auto'):
        if method == 'auto':
            try:
                import pyopt
                method = 'pyOpt.SLSQP'
            except ImportError:
                method = 'scipy'

        self.method = method
        self.seed = None

    def setSeed(self,x):
        self.seed = x

    def solve(self,problem,numIters=100,tol=1e-6):
        """Returns a tuple (success,result)"""
        if self.seed is None:
            raise RuntimeError("Need to provide a seed state")
        if problem.objective is None:
            raise RuntimeError("Need to provide an objective function")
        if self.method.startswith('scipy'):
            from scipy import optimize
            items = self.method.split('.')
            scipyMethod = 'SLSQP'
            if len(items)>1:
                scipyMethod = items[1]
            jac = False
            if problem.objectiveGrad:
                jac = problem.objectiveGrad
            bounds = None
            if problem.bounds:
                bounds = zip(*problem.bounds)
            constraintDicts = []
            for i in xrange(len(problem.equalities)):
                constraintDicts.append({'type':'eq','fun':problem.equalities[i]})
                if problem.equalityGrads[i] is not None:
                    constraintDicts[-1]['jac'] = problem.equalityGrads[i]
            for i in xrange(len(problem.inequalities)):
                constraintDicts.append({'type':'ineq','fun':problem.inequalities[i]})
                if problem.inequalityGrads[i] is not None:
                    constraintDicts[-1]['jac'] = problem.inequalityGrads[i]
            res = optimize.minimize(problem.objective,x0=self.seed,method=scipyMethod,
                                    jac=jac,bounds=bounds,
                                    constraints=constraintDicts,tol=tol,options={'maxiter':numIters})
            return res.success,res.x.tolist()
        elif self.method.startswith('pyOpt'):
            import pyOpt
            import warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning) 
            items = self.method.split('.')
            pyOptMethod = 'SLSQP'
            if len(items)>1:
                pyOptMethod = items[1]

            def objfunc(x):
                fx = problem.objective(x)
                gx = sum([f(x) for f in problem.equalities]+[f(x) for f in problem.inequalities],[])
                gx = gx + (x-problem.bounds[1]).tolist() + (problem.bounds[0]-x).tolist()
                #flag = True if problem.feasibilityTest is None else problem.feasibilityTest(x)
                flag = True
                return fx,gx,flag
            opt_prob = pyOpt.Optimization('',objfunc)
            opt_prob.addObj('f')
            for i in range(len(self.seed)):
                opt_prob.addVar('x'+str(i),'c',lower=problem.bounds[0][i],upper=problem.bounds[1][i],value=self.seed[i])
            hlen = sum(len(f(self.seed)) for f in problem.equalities)
            glen = sum(len(f(self.seed)) for f in problem.inequalities)
            opt_prob.addConGroup('eq',hlen,'e')
            opt_prob.addConGroup('ineq',glen,'i')
            #expressing bounds as inequalities
            opt_prob.addConGroup('bnd',len(self.seed)*2,'i')

            opt = getattr(pyOpt,pyOptMethod)()
            opt.setOption('IPRINT', -1)
            opt.setOption('MAXIT',numIters)
            opt.setOption('ACC',tol)
            sens_type = 'FD'
            if problem.objectiveGrad is not None:
                #user provided gradients
                if all(f is not None for f in problem.equalityGrads) and all(f is not None for f in problem.inequalityGrads):
                    def objfuncgrad(x):
                        fx = problem.objectiveGrad(x)
                        gx = sum([f(x) for f in problem.equalityGrads]+[f(x) for f in problem.equalityGrads],[])
                        for i in range(len(x)):
                            zero = [0]*len(x)
                            zero[i] = 1
                            gx.append(zero)
                        for i in range(len(x)):
                            zero = [0]*len(x)
                            zero[i] = -1
                            gx.append(zero)
                        flag = True
                        return fx,gx,flag
                    #TEMP: test no analytic gradients
                    #sens_type = objfuncgrad
                else:
                    print "Warning, need all or no gradients provided"
            [fstr, xstr, inform] = opt(opt_prob,sens_type=sens_type)
            if inform['value'] != 0:
                return False,xstr.tolist()
            f,g,flag = objfunc(xstr)
            #flag doesn't check?
            eqfeasible = all(abs(v)<tol for v in g[:hlen])
            ineqfeasible = all(v <= 0 for v in g[hlen:hlen+glen])
            boundfeasible = all(a<=x and x<=b for x,a,b in zip(xstr,problem.bounds[0],problem.bounds[1]))
            feasible = eqfeasible and ineqfeasible and boundfeasible
            if not feasible:
                if not boundfeasible:
                    #try clamping
                    for i in xrange(len(xstr)):
                        xstr[i] = min(max(xstr[i],problem.bounds[0][i]),problem.bounds[1][i])
                    f,g,flag = objfunc(xstr)
                    boundfeasible = True
                    eqfeasible = all(abs(v)<tol for v in g[:hlen])
                    ineqfeasible = all(v <= 0 for v in g[hlen:hlen+glen])
                    feasible = eqfeasible and ineqfeasible and boundfeasible
                """if not feasible:
                    print "Strange, optimizer says successful and came up with an infeasible solution",eqfeasible,ineqfeasible,boundfeasible
                    print fstr,xstr,inform
                    print problem.equalities[0](xstr)
                    print g
                """
            return feasible,xstr.tolist()
        else:
            raise RuntimeError('Invalid method specified: '+self.method)

class GlobalOptimizer:
    """A wrapper around different local optimization libraries. Only
    minimization is supported, and only scipy and pyOpt are supported.
    
    The method is specified using the method string, which can be:
    - auto: picks between DIRECT and random-restart
    - random-restart: random restarts using 
    - DIRECT: the DIRECT method
    - scipy: uses scipy.optimize.minimize with default settings.
    - scipy.[METHOD]: uses scipy.optimize.minimize with the argument
      method=[METHOD].
    - pyOpt: uses pyOpt with SLSQP.
    - pyOpt.[METHOD]: uses pyOpt with the given method.

    The method attribute can also be a list, which does a cascading solver
    in which the previous solution point is used as a seed for the next
    solver.
    """
    def __init__(self,method='auto'):
        if method == 'auto':
            #Runs the DIRECT method
            #method = 'DIRECT'
            #Runs the DIRECT method then cleans it up with the default local
            #optimizer
            #method = ['DIRECT','auto']
            #Runs the scipy differential evolution technique
            method = 'scipy.differential_evolution'
            #Runs random restarts using scipy as a local optimizer
            #method = 'random-restart.scipy'
            #Runs random restarts using pyOpt as a local optimizer
            #method = 'random-restart.pyOpt.SLSQP'
        self.method = method
        self.seed = None

    def setSeed(self,x):
        self.seed = x

    def solve(self,problem,numIters=100,tol=1e-6):
        """Returns a pair (solved,x) where solved is True if the solver
        found a valid solution, and x is the solution vector."""
        if isinstance(self.method,(list,tuple)):
            #sequential solve
            seed = self.seed
            for i,m in enumerate(self.method):
                if hasattr(numIters,'__iter__'):
                    itersi = numIters[i]
                else:
                    itersi = numIters
                if hasattr(tol,'__iter__'):
                    toli = tol[i]
                else:
                    toli = tol
                print "Step",i,"method",m,'iters',itersi,'tol',toli
                if m == 'auto':
                    opt = LocalOptimizer(m)
                else:
                    opt = GlobalOptimizer(m)
                #seed with previous seed, if necessary
                opt.setSeed(seed)
                (succ,xsol)=opt.solve(problem,itersi,toli)
                if not succ: return (False,xsol)
                seed = xsol[:]
            return ((seed is not None),seed)
        elif self.method == 'scipy.differential_evolution':
            from scipy import optimize
            if problem.bounds == None:
                raise RuntimeError("Cannot use scipy differential_evolution method without a bounded search space")
            flattenedProblem = problem.flatten(objective_scale = 1e-5)
            res = optimize.differential_evolution(flattenedProblem.objective,zip(*flattenedProblem.bounds))
            print "scipy.differential_evolution solution:",res.x
            print "Objective value",res.fun
            print "Equality error:",[gx(res.x) for gx in problem.equalities]
            return (True,res.x)
        elif self.method == 'DIRECT':
            import DIRECT
            if problem.bounds == None:
                raise RuntimeError("Cannot use DIRECT method without a bounded search space")
            flattenedProblem = problem.flatten(objective_scale = 1e-5)
            minval = [float('inf'),None]
            def objfunc(x,userdata):
                v = flattenedProblem.objective(x)
                if v < userdata[0]:
                    userdata[0] = v
                    userdata[1] = [float(xi) for xi in x]
                return v
            (x,fmin,ierror)=DIRECT.solve(objfunc,problem.bounds[0],problem.bounds[1],eps=tol,maxT=numIters,maxf=40000,algmethod=1,user_data=minval)
            print "DIRECT solution:",x
            print "Objective value",fmin
            print "Minimum value",minval[0],minval[1]
            print "Error:",ierror
            print "Equality error:",[gx(x) for gx in problem.equalities]
            return (True,minval[1])
        elif self.method.startswith('random-restart'):
            import random
            if problem.bounds == None:
                raise RuntimeError("Cannot use random-restart method without a bounded search space")
            localmethod = self.method[15:]
            lopt = LocalOptimizer(localmethod)
            best = self.seed
            fbest = (problem.objective(best) if best is not None else float('inf'))
            for it in xrange(numIters[0]):
                x = [random.uniform(a,b) for a,b in zip(*problem.bounds)]
                lopt.setSeed(x)
                succ,x = lopt.solve(problem,numIters[1],tol)
                if succ:
                    fx = problem.objective(x)
                    if fx < fbest:
                        fbest = fx
                        best = x
            return (best is not None, best)
        else:
            opt = LocalOptimizer(self.method)
            opt.setSeed(self.seed)
            return opt.solve(problem,numIters,tol)
