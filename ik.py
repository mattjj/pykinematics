from __future__ import division
import numpy as np
import abc, itertools, hashlib, operator
from numpy.core.umath_tests import inner1d

from util import rot3D_YawPitchRoll, solve_psd2, block_view

#######################
#  Geometric Algebra  #
#######################

class GeneralLinearElement(object):
    '''
    models an element of GL(n+1)
    '''
    def __init__(self,A):
        self._mat = A

    def __mul__(self,other):
        new = self.__class__.__new__()
        new.A = self._mat.dot(other._mat)

    def apply(self,vec):
        return self._mat.dot(vec)


class AffineElement(GeneralLinearElement):
    '''
    models an element of the affine group (as a subgroup of GL(n+1)) with an
    action on R^n defined in homogeneous coordinates
    A(x+b), NOT Ax+b !!!
    '''
    def __init__(self,A,b):
        self.set_matrices(A,b)

    def set_matrices(self,A,b):
        n = b.shape[0]
        self._mat = np.zeros((n+1,n+1))
        self._mat[:-1,:-1] = A
        self._mat[:-1,-1] = A.dot(b)
        self._mat[-1,-1] = 1
        return self


class SpecialEuclideanElement(AffineElement):
    '''
    models an element of E+(n), where E+(n) / T(n) = SO(n)
    a subgroup of affine transformations where the non-translation part is in SO(n)
    '''
    def __init__(self,rotation_matrix,translation):
        self.set_matrices(rotation_matrix,translation)

    def set_matrices(self,rotation_matrix,translation):
        return super(SpecialEuclideanElement,self).set_matrices(A=rotation_matrix,b=translation)


class SpecialEuclideanLieAlgebraElement(GeneralLinearElement):
    '''
    models an element of the Lie algebra of E+(n) as an element of GL(n+1).
    the GL(n) part is skew-symmetric
    by applying the action of the Lie group we get a section of the tangent bundle
    '''
    def __init__(self,skew_symmetric_matrix,inftranslation):
        self.set_matrices(skew_symmetric_matrix,inftranslation)

    def set_matrices(self,skew_symmetric_matrix,inftranslation):
        n = inftranslation.shape[0]
        self._mat = np.zeros((n+1,n+1))
        self._mat[:-1,:-1] = skew_symmetric_matrix
        self._mat[:-1,-1] = skew_symmetric_matrix.dot(inftranslation)
        return self

# charts (parameterizations) let us specify transformations via angles and do
# calculus with respect to those angle coordinates

class ChartedSpecialEuclideanElement(SpecialEuclideanElement):
    'models an element of E+(n) along with a chart (for some submanifold)'
    __metaclass__ = abc.ABCMeta

    # also requires a deriv_tensor_at_identity member

    @abc.abstractmethod
    def set_from_chart(self,coordinates):
        pass

# here are some concrete charts

class RotorJoint2D(ChartedSpecialEuclideanElement):
    '''
    models a rotor joint in 2D with fixed length (as an element of a charted
    submanifold of E+(2))
    '''

    numcoordinates = 1
    ndim = 2

    deriv_tensor_at_identity = np.array([SpecialEuclideanLieAlgebraElement(
        skew_symmetric_matrix=np.array(( (0.,-1.), (1.,0.) )),
        inftranslation=np.zeros(2))._mat])

    def __init__(self,length,theta=None):
        self.l = length
        if theta is not None:
            self.set_from_chart(theta)

    def set_from_chart(self,theta):
        c, s = np.cos(theta), np.sin(theta)
        self._mat = np.array(( (c,-s, c*self.l),
                               (s, c, s*self.l),
                               (0, 0, 1       ), ))

# TODO test
class RotorJoint3DYawPitchRoll(ChartedSpecialEuclideanElement):
    '''
    models a rotor joint in 3D with fixed length (as an element of a charted
    submanifold of E+(3)) using the Yaw-Pitch-Roll chart
    '''

    numcoordinates = 3
    ndim = 3

    deriv_tensor_at_identity = np.array([SpecialEuclideanLieAlgebraElement(skew_symmetric_matrix=m,
                                      inftranslation=np.zeros(3))._mat \
                        for m in \
                            [np.array(( ( 0.,-1., 0.),
                                        ( 1., 0., 0.),
                                        ( 0., 0., 0.) )),
                            np.array((  ( 0., 0.,-1.),
                                        ( 0., 0., 0.),
                                        ( 1., 0., 0.) )),
                            np.array((  ( 0., 0., 0.),
                                        ( 0., 0.,-1.),
                                        ( 0., 1., 0.) ))]])

    def __init__(self,length,theta_x=None,theta_y=None,theta_z=None):
        self.length = length
        if all(x is not None for x in (theta_x,theta_y,theta_z)):
            self.set_from_chart(theta_x,theta_y,theta_z)

    def set_from_chart(self,theta_x,theta_y,theta_z):
        return self.set_matrices(rotation_matrix=rot3D_YawPitchRoll(theta_x,theta_y,theta_z),
                                 translation=np.array((self.length,0,0)))

########
#  FK  #
########

class ForwardKinematics(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self,coordinates):
        pass

    @abc.abstractmethod
    def deriv(self,coordinates):
        pass

### FK trees are hard! or maybe this can be improved...

class JointTreeNode(object):
    def __init__(self,E,effectors=[],children=[]):
        self.E = E
        self.effectors = [np.concatenate((e,(1.,))) for e in effectors]
        self.children = children

class JointTreeFK(ForwardKinematics):

    ### initialization stuff

    def __init__(self,root):
        self.root = root
        self.ndim = self.root.E.ndim
        self.coordinate_nums = []

        node_idx, eff_idx = itertools.count(), itertools.count()
        self._set_indices(root,node_idx,eff_idx)
        self.num_joints, self.num_effectors = node_idx.next(), eff_idx.next()

        self._prev_call_coord_hash = None
        self._prev_call_result = None
        self._J = np.zeros((self.num_effectors*self.ndim, max(self.coordinate_nums)*self.num_joints))
        self._Jblocks = block_view(self._J,(self.ndim,max(self.coordinate_nums)))

    def _set_indices(self,node,node_indexer,effector_indexer):
        # nodes are indexed by left-to-right depth-first search preorder
        node.idx = node_indexer.next()
        node.effectors = [(effector_indexer.next(), e) for e in node.effectors]
        self.coordinate_nums.append(node.E.numcoordinates)
        for c in node.children:
            self._set_indices(c,node_indexer,effector_indexer)

    ### implementing the FK interface

    def __call__(self,coordinates):
        h = hashlib.sha1(coordinates).hexdigest()
        if not (self._prev_call_coord_hash == h):
            self._prev_call_coord_hash = h
            self._set_coordinates(coordinates,self.root)
            self._prev_call_result = [e for i,e in self._get_effectors(self.root)]
        return np.asarray(self._prev_call_result)[:,:-1]

    def deriv(self,coordinates):
        self(coordinates)
        # sorting slightly improves memory access pattern and is a tiny net
        # performance gain. python sorts tuples lexicographically. python's
        # heapq.merge (and heaps in general) doesn't support a key argument and
        # sorting at the top level was slightly better than sorting at each leve
        # of the recursion despite Timsort looking for sorted sublists
        for (effidx,jointidx), d in sorted(self._get_derivatives(self.root),key=operator.itemgetter(0)):
            self._Jblocks[effidx,jointidx] = d[:-1,:]
        return self._J

    ### internal computation

    def _set_coordinates(self,coordinates,node):
        node.E.set_from_chart(coordinates[node.idx])
        for c in node.children:
            self._set_coordinates(coordinates,c)

    def _get_effectors(self,node):
        node.localized_effectors = [(idx,node.E.apply(e)) for idx,e in
                itertools.chain(
                    node.effectors,
                    (epair for c in node.children for epair in self._get_effectors(c)))]
        return node.localized_effectors

    def _get_derivatives(self,node):
        return itertools.chain(
                (((effidx,jointidx), node.E.apply(ds))
                    for c in node.children for (effidx,jointidx), ds in self._get_derivatives(c)),
                (((effidx,node.idx), np.tensordot(node.E.deriv_tensor_at_identity,eff,1).T)
                    for effidx, eff in node.localized_effectors),
                )


class BigchartJointTreeFK(JointTreeFK):
    def __init__(self,root,bigchart,allmaps):
        super(BigchartJointTreeFK,self).__init__(root)
        self.bigchart = bigchart
        self.allmaps = allmaps
        for i,n in enumerate(self._tree_dfs(root)):
            n.E._mat = allmaps[i]

    def _tree_dfs(self,node):
        yield node
        for c in node.children:
            for node in self._tree_dfs(c):
                yield node

    def _set_coordinates(self,coordinates,_):
        self.bigchart(coordinates,self.allmaps)

########
#  IK  #
########

def construct_solver(s,dampening_factors,tol,maxiter,
        prior_variances=None,jointmins=-np.inf,jointmaxes=np.inf,errorlimits=(-np.inf,np.inf)):

    if prior_variances is None:
        prior_variances = np.ones(s._J.shape[1])
    else:
        prior_variances = np.sqrt(prior_variances) # really std devs

    def solver(t,theta_init):
        theta = np.array(theta_init,copy=True)
        e = np.clip(t-s(theta),*errorlimits)
        JJT = np.empty((t.size,t.size))
        for itr in range(maxiter):
            J = s.deriv(theta) * prior_variances
            JJT = np.dot(J,J.T,out=JJT)
            JJT.flat[::JJT.shape[0]+1] += dampening_factors
            theta.flat += J.T.dot(solve_psd2(JJT,e.ravel(),overwrite_b=True))
            np.clip(theta,jointmins,jointmaxes,out=theta)
            e = np.clip(t-s(theta),*errorlimits,out=e)
            if inner1d(e,e).max() < tol**2:
                return theta
        return theta

    return solver

# TODO test 3D
# TODO benchmarking code
