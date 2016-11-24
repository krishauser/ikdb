import pkg_resources
if pkg_resources.get_distribution('klampt').version >= '0.7':
    from klampt.math import vectorops
    from klampt.model import collide
else:
    from klampt import vectorops
    from klampt import robotcollide as collide
import numpy as np

_functionFactories = dict()

def mahalanobis_distance2(u,v,A=None):
    if A is None:
        return vectorops.distanceSquared(u,v)
    else:
        z = np.array(vectorops.sub(u,v))
        return np.dot(z.T,np.dot(A,z))

def registerFunctionFactory(type,makeFunction):
    """Registers a function factory.  The function makeFunction takes an
    argument bundle and returns a function of a single parameter."""
    global _functionFactories
    _functionFactories[type] = makeFunction

def registerFunction(type,f,varname='x'):
    """Registers a function that can take multiple arguments, including the
    variable x.  The factory is constructed so that it can be passed a
    dictionary of named arguments to f, or a list of arguments (not equal to
    x), or 0 or 1 arguments.
    """
    import inspect
    (args,varargs,keywords,defaults) = inspect.getargspec(f)
    if varname not in args:
        raise ValueError("The variable name "+varname+" must be in the function's argment list, instead got "+",".join(args))
    if varargs is not None:
        print "registerFunction: Warning, may have errors with variable arguments"
    argindex = args.index(varname)
    def makefunc(argbundle):
        if isinstance(argbundle,dict):
            assert varname not in argbundle,"Must not pass value "+varname+" to function "+func
            return lambda x:f(varname=x,**argbundle)
        elif isinstance(argbundle,(tuple,list)):
            assert len(args)==len(argbundle)+1,"Must pass in all other arguments "+",".join(args)+" except for "+varname+" to function "+func
            #need to pass in order
            return lambda x:f(*(argbundle[:argindex]+[x]+argbundle[argindex:]))
        elif argbundle is not None:
            assert len(args)==2,"Not enough arguments passed to function "+func
            if argindex==0:
                return lambda x:f(x,argbundle)
            elif argindex==1:
                return lambda x:f(x,argbundle)
        else:
            assert len(args)==1,"Not enough arguments passed to function "+func
            return lambda x:f(x)
    registerFunctionFactory(type,makefunc)

def linear(x,c0,c1):
    return np.dot(c1,x)+c0

def quadratic(x,c0,c1,c2):
    return np.dot(x.T,np.dot(c2,x))+np.dot(c1,x)+c0

def distance_L2(x,center):
    return np.linalg.norm(np.array(center)-x)

def distance_L1(x,center):
    return np.linalg.norm(np.array(center)-x,1)

def distance_Linf(x,center):
    return np.linalg.norm(np.array(center)-x,np.inf)

def distance_squared_L2(x,center):
    return np.linalg.norm(np.array(center)-x)

def andFunction(*fs):
    return lambda x: all(f(x) for f in fs)

def registerDefaultFunctions():
    registerFunction('constant',lambda x,value:value)
    registerFunction('linear',linear)
    registerFunction('quadratic',quadratic)
    registerFunction('distance',distance_L2)
    registerFunction('distance_L2',distance_L2)
    registerFunction('distance_L1',distance_L1)
    registerFunction('distance_Linf',distance_Linf)
    registerFunction('distanceSquared',distance_squared_L2)

def registerCollisionFunction(world,name='collisionFree'):
    collider = collide.WorldCollider(world)
    robot = world.robot(0)
    def collides(x):
        robot.setConfig(x)
        for c in collider.collisions():
            return False
        return True
    registerFunctionFactory(name,lambda args: collides)

def registerJointRangeCostFunction(robot,name='jointRangeCost'):
    def jointRangeCost(x,qmin,qmax):
        d = 0
        for xi,a,b in zip(x,qmin,qmax):
            if a==b: continue
            d += (min(xi-a,b-xi)/(b-a))**2
        return -d
    qmin,qmax = robot.getJointLimits()
    registerFunctionFactory(name,lambda args:(lambda x:jointRangeCost(x,qmin,qmax)))
    registerFunction(name+'_dynamic',jointRangeCost)

def makeFunction(type,arguments):
    """Makes a function from a dict object, given the current function registry.
    Returns None on failure."""
    global _functionFactories
    try:
        f = _functionFactories[type]
    except KeyError:
        print "Function of type",type,"does not exist"
        return None
    return f(arguments)
