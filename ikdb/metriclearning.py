import numpy as np
import math

def mahalanobis_distance(u,v,A):
    return math.sqrt(mahalanobis_distance2(u,v,A))

def mahalanobis_distance2(u,v,A):
    z = u-v
    if A is None:
        return np.dot(z,z)
    return np.dot(z.T,np.dot(A,z))


def metric_logdet_update(A,u,v,desiredDistance,sign,regularizationParameter):
    """Given points u and v which violate a Mahalanobis distance constraint
       sign*d_A(u,v) = sign*(u-v)^T A (u-v) <= y,
    takes a step to analytically solve the regularized log-det problem
    A' = argmin D(A,A') + nu*loss(d_A'(u,v),y)
    with nu = regularizationParameter, y = desiredDistance (usually 1), and sign=+/- 1.

    Returns the new Mahalanobis distance matrix A'.

    A, u, and v are numpy arrays of dimension 2, 1, 1, respectively.

    Method is from [Jain 2008].
    """
    nu = regularizationParameter
    z = u - v
    y = desiredDistance
    Az = np.dot(A,z)
    ypred = np.dot(z.T,Az)
    if sign*(ypred - desiredDistance) <= 0:
        return A
    if ypred < 1e-7: return A
    #c = nu*y*ypred-1
    #yest = (c + math.sqrt(c*c + 4*nu*ypred*ypred))/(2*nu*ypred)
    c = nu*y - 1.0/ypred
    yest = (c + math.sqrt(c*c + 4*nu))/(2*nu)
    if sign*(yest - desiredDistance) <= 0:
        return A
    scale = (nu*(yest-y))/(1+nu*(yest-y)*ypred)
    return A - scale*np.outer(Az,Az)

def metric_logdet_learn(examples,regularizationParameter):
    """Given a list of examples (u,v,sign) where u and v are vectors
    and sign is +/- 1 depending on whether they should be close (+1) or
    far (-1), learns a metric matrix A such that
    (x-y)^T A (x-y) <= 1 signifies x and y should be close, and >1
    signifies they should be far. """
    A = np.eye(len(examples[0][0]))
    for (u,v,sign) in examples:
        A = metric_logdet_update(A,u,v,1,regularizationParameter)
    return A

def cholesky_update(L,x):
    x = x[:]  #make a copy, we'll change it
    n = L.shape[0]
    assert n == L.shape[1] and n == len(x)
    alpha=1
    for i in xrange(n):
        deltai = pow(L[i,i],2);
        temp = alpha + pow(x[i],2)/deltai
        deltai = deltai*temp
        gamma = x[i]/deltai
        deltai = deltai / alpha
        alpha = temp
        L[i,i] = math.sqrt(deltai)
        for k in xrange(i+1,n):
            x[k] -= x[i]*L[k,i]
            L[k,i] += gamma*x[k]

def cholesky_downdate(L,x):
    x = x[:]
    n = L.shape[0]
    assert n == L.shape[1] and n == len(x)
    alpha=1
    for i in xrange(n):
        deltai = pow(L[i,i],2);
        temp = alpha - pow(x[i],2)/deltai
        deltai = deltai*temp
        if deltai == 0:
            raise RuntimeError("Matrix got a zero on the diagonal entry %d"%(i,))
        gamma = x[i]/deltai
        deltai = deltai / alpha
        alpha = temp
        if deltai < 0:
            raise RuntimeError("Matrix became negative definite in entry %d"%(i,))
        L[i,i] = math.sqrt(deltai)
        for k in xrange(i+1,n):
            x[k] -= x[i]*L[k,i]
            L[k,i] -= gamma*x[k]

def metric_logdet_update_cholesky(L,u,v,desiredDistance,sign,regularizationParameter):
    """Given points u and v which violate a Mahalanobis distance constraint
       sign*d_A(u,v) = sign*(u-v)^T A (u-v) <= y,
    with A = LL^T, a Cholesky factorization,
    takes a step to analytically solve the regularized log-det problem
    A' = argmin D(A,A') + nu*loss(d_A'(u,v),y)
    with nu = regularizationParameter, y = desiredDistance, sign = 1 or -1.

    L is modified to be the Cholesky decomposition L' of the new Mahalanobis
    distance matrix A' s.t. A' = L' L'^T.

    L, u, and v are numpy arrays of dimension 2, 1, 1, respectively.
    """
    nu = regularizationParameter
    z = u - v
    y = desiredDistance
    Ltz = np.dot(L.T,z)
    Az = np.dot(L,Ltz)
    ypred = np.dot(Ltz.T,Ltz)
    if sign*(ypred-desiredDistance) <= 0: return 
    if ypred < 1e-7: return
    c = nu*y-1/ypred
    yest = (c + math.sqrt(c*c + 4*nu))/(2*nu)
    if sign*(yest - desiredDistance) <= 0: return 
    scale = nu*(yest-y)/(1+nu*(yest-y)*ypred)
    if scale < 0:
        update = Az * math.sqrt(-scale)
        cholesky_update(L,update)
    else:
        downdate = Az * math.sqrt(scale)
        cholesky_downdate(L,downdate)

if __name__ == '__main__':
    import time
    Atrue = np.eye(2)
    Atrue[0,0] = 5.0
    Ltrue = np.linalg.cholesky(Atrue)
    thresh = 1.0
    A = np.eye(2)
    L = np.eye(2)
    numpts = 10000
    regularization = 1.0
    t0 = time.time()
    for i in range(numpts):
        pt1 = [np.random.uniform(-10,10),np.random.uniform(-10,10)]
        pt2 = [pt1[0]+np.random.uniform(-2,2),pt1[1]+np.random.uniform(-2,2)]
        pt1 = np.array(pt1)
        pt2 = np.array(pt2)
        dtrue = mahalanobis_distance(pt1,pt2,Atrue)
        A = metric_logdet_update(A,pt1,pt2,thresh,(1 if dtrue < thresh else -1),regularization)
        metric_logdet_update_cholesky(L,pt1,pt2,thresh,(1 if dtrue < thresh else -1),regularization)
        #A = np.dot(L,L.T)
        #print "Update",i,":",A
    t1 = time.time()
    print numpts,"updates, time",t1-t0
    print "Result",A
    print "Result cholesky",L
