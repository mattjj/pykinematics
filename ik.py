from __future__ import division
import numpy as np
import abc, itertools
from numpy.core.umath_tests import inner1d

from util import rot2D, rot3D_YawPitchRoll, solve_psd, flatten1

#######################
#  Geometric Algebra  #
#######################

class GeneralLinearElement(object):
    '''
    models an element of GL(n+1) with a slightly unusual action on R^n
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
    models an element of the affine group (as a subgroup of GL(n+1))
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
    models an element of the tangent bundle of E+(n) as an element of GL(n+1).
    the GL(n) part is skew-symmetric
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

    @abc.abstractmethod
    def set_from_chart(self,coordinates):
        pass

    @abc.abstractmethod
    def tangent_basis_at_identity(self):
        pass

# here are some concrete charts

class RotorJoint2D(ChartedSpecialEuclideanElement):
    '''
    models a rotor joint in 2D with fixed length (as an element of a charted
    submanifold of E+(2))
    '''

    numcoordinates = 1
    ndim = 2

    tangent_basis = [SpecialEuclideanLieAlgebraElement(
        skew_symmetric_matrix=np.array(( (0.,-1.), (1.,0.) )),
        inftranslation=np.zeros(2))]

    def __init__(self,length,theta=None):
        self.length = length
        if theta is not None:
            self.set_from_chart(theta)

    def set_from_chart(self,theta):
        return self.set_matrices(rotation_matrix=rot2D(theta),translation=np.array((self.length,0)))

    def tangent_basis_at_identity(self):
        return self.tangent_basis


# TODO test
class RotorJoint3DYawPitchRoll(ChartedSpecialEuclideanElement):
    '''
    models a rotor joint in 3D with fixed length (as an element of a charted
    submanifold of E+(3)) using the Yaw-Pitch-Roll chart
    '''

    numcoordinates = 3
    ndim = 3

    tangent_basis = [SpecialEuclideanLieAlgebraElement(skew_symmetric_matrix=m,
                                      inftranslation=np.zeros(3)) \
                        for m in \
                            [np.array(( ( 0.,-1., 0.),
                                        ( 1., 0., 0.),
                                        ( 0., 0., 0.) )),
                            np.array((  ( 0., 0.,-1.),
                                        ( 0., 0., 0.),
                                        ( 1., 0., 0.) )),
                            np.array((  ( 0., 0., 0.),
                                        ( 0., 0.,-1.),
                                        ( 0., 1., 0.) ))]]

    def __init__(self,length,theta_x=None,theta_y=None,theta_z=None):
        self.length = length
        if all(x is not None for x in (theta_x,theta_y,theta_z)):
            self.set_from_chart(theta_x,theta_y,theta_z)

    def set_from_chart(self,theta_x,theta_y,theta_z):
        return self.set_matrices(rotation_matrix=rot3D_YawPitchRoll(theta_x,theta_y,theta_z),
                                 translation=np.array((self.length,0,0)))

    def tangent_basis_at_identity(self):
        return self.tangent_basis

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

### FK chains are nice

class JointChainFK(ForwardKinematics):
    pass # TODO

### FK trees are hard! or maybe this can be improved...

class JointTreeNode(object):
    def __init__(self,E,effectors=[],children=[]):
        self.E = E
        self.effectors = [np.concatenate((e,(1.,))) for e in effectors]
        self.children = children

class JointTreeFK(ForwardKinematics):
    # nodes are indexed by left-to-right depth-first search preorder
    def __init__(self,root):
        self.root = root
        self.ndim = self.root.E.ndim
        self.coordinate_nums = []

        node_idx, eff_idx = itertools.count(), itertools.count()
        self._set_indices(root,node_idx,eff_idx)
        self.num_joints, self.num_effectors = node_idx.next(), eff_idx.next()

    def __call__(self,coordinates):
        self.coordinates = coordinates
        self._set_coordinates(coordinates,self.root)
        return np.asarray(self._get_effectors(self.root))[:,:-1]

    def deriv(self,coordinates):
        self._set_coordinates(coordinates,self.root)
        J = np.zeros((self.num_effectors, self.ndim, max(self.coordinate_nums), self.num_joints))
        for (jointidx,effidx), d in self._get_derivatives(self.root)[1]:
            J[effidx,:,:len(d),jointidx] = np.asarray(d)[:,:-1].T
        return J

    def _set_indices(self,node,node_indexer,effector_indexer):
        node.idx = node_indexer.next()
        node.effector_indices = [effector_indexer.next() for e in node.effectors]
        self.coordinate_nums.append(node.E.numcoordinates)
        for c in node.children:
            self._set_indices(c,node_indexer,effector_indexer)

    def _set_coordinates(self,coordinates,node):
        node.E.set_from_chart(coordinates[node.idx])
        for c in node.children:
            self._set_coordinates(coordinates,c)

    def _get_effectors(self,node):
        return map(node.E.apply, node.effectors
                + flatten1([self._get_effectors(c) for c in node.children]))

    def _get_derivatives(self,node):
        effectors, derivs = map(flatten1,zip(*[self._get_derivatives(c) for c in node.children])) \
                if len(node.children) > 0 else ([], [])
        effectors = [(effidx, node.E.apply(eff)) for effidx, eff in \
                itertools.chain(effectors,zip(*(node.effector_indices,node.effectors)))]
        derivs = [((jointidx,effidx), map(node.E.apply, deriv)) for (jointidx, effidx), deriv in derivs] \
                  + [((node.idx,effidx), [d.apply(eff) for d in node.E.tangent_basis_at_identity()])
                        for effidx, eff in effectors]
        return effectors, derivs

########
#  IK  #
########

def construct_solver(s,dampening_factors,tol,maxiter,limits=(-np.inf,np.inf)):
    def solver(t,theta_init):
        theta = np.array(theta_init,copy=True)
        e = np.clip(t-s(theta),*limits)
        for itr in range(maxiter):
            J = s.deriv(theta).reshape((-1,theta.size))
            JJT = J.dot(J.T)
            JJT.flat[::JJT.shape[0]+1] += dampening_factors
            theta.flat += J.T.dot(solve_psd(JJT,e.ravel(),overwrite_b=True))
            e = np.clip(t-s(theta),*limits,out=e)
            if inner1d(e,e).max() < tol**2:
                return theta
        return theta
    return solver

# TODO test 3D

