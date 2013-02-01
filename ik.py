from __future__ import division
import numpy as np
import abc

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
        return self._mat[:-1,:-1].dot(vec + self._mat[:-1,-1])


class AffineElement(GeneralLinearElement):
    '''
    models an element of the affine group (as a subgroup of GL(n+1))
    '''
    def __init__(self,A,b):
        self.set_matrices(A,b)

    def set_matrices(self,A,b):
        self.ndim = n = b.shape[0]
        self._mat = np.zeros((n+1,n+1))
        self._mat[:-1,:-1] = A
        self._mat[:-1,-1] = b
        self._mat[-1,-1] = 1
        return self

    def apply_to_tangentvec(self,vec):
        return self._mat[:-1,:-1].dot(vec)


class SpecialEuclideanElement(AffineElement):
    '''
    models an element of E+(n), where E+(n) / T(n) = SO(n)
    a subgroup of affine transformations where the non-translation part is in SO(n)
    '''
    def __init__(self,rotation_matrix,translation):
        self.set_matrices(rotation_matrix,translation)

    def set_matrices(self,rotation_matrix,translation):
        return super(SpecialEuclideanElement,self).set_matrices(A=rotation_matrix,b=translation)


class SpecialEuclideanTangentElement(GeneralLinearElement):
    '''
    models an element of the tangent bundle (Lie algebra) of E+(n)
    an element of GL(n+1). the GL(n) part is skew-symmetric
    '''
    def __init__(self,skew_symmetric_matrix,inftranslation):
        self.set_matrices(skew_symmetric_matrix,inftranslation)

    def set_matrices(self,skew_symmetric_matrix,inftranslation):
        self.ndim = n = inftranslation.shape[0]
        self._mat = np.zeros((n+1,n+1))
        self._mat[:-1,:-1] = skew_symmetric_matrix
        self._mat[:-1,-1] = inftranslation
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

# here are some concrete charted maps

class RotorJoint2D(ChartedSpecialEuclideanElement):
    '''
    models a rotor joint in 2D with fixed length (as an element of a charted
    submanifold of E+(2))
    '''
    tangent_basis = [SpecialEuclideanTangentElement(
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

    tangent_basis = [SpecialEuclideanTangentElement(skew_symmetric_matrix=m,
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


class JointTreeNode(object):
    def __init__(self,idx,E,effectors=[],children=[]):
        self.idx = idx # TODO indices should be internal,right?
        self.E = E
        self.effectors = effectors
        self.children = children

    def set(self,coordinates):
        self.coordinates = coordinates
        self.E.set_from_chart(coordinates[self.idx])
        for c in self.children:
            c.set(coordinates)

    def get_effectors(self):
        return map(self.E.apply,
                self.effectors + flatten1([c.get_effectors() for c in self.children]))

    def get_derivatives(self):
        # TODO TODO totally busted for more effectors! need some tree to matrix
        # business!
        effectors, derivs = map(flatten1,zip(*[c.get_derivatives() for c in self.children])) \
                if len(self.children) > 0 else ([], [])
        effectors = map(self.E.apply, self.effectors + effectors)
        derivs = [d.apply(e) for e in effectors for d in self.E.tangent_basis_at_identity()] \
                + map(self.E.apply_to_tangentvec,derivs)
        return effectors, derivs


class JointTreeFK(ForwardKinematics):
    def __init__(self,tree):
        self.tree = tree

    def __call__(self,coordinates):
        self.tree.set(coordinates)
        return np.asarray(self.tree.get_effectors()).ravel()

    def deriv(self,coordinates):
        self.tree.set(coordinates)
        effectors, derivs = self.tree.get_derivatives()
        return np.asarray(derivs).T

### special cases

class JointChain2DFK(ForwardKinematics):
    pass # TODO

class JointChain3DFK(ForwardKinematics):
    pass # TODO

########
#  IK  #
########

def construct_solver(s,dampening_factors,tol=1e-2,maxiter=2000):
    def solver(t,theta_init):
        theta = np.array(theta_init,copy=True)
        for itr in range(maxiter):
            J = s.deriv(theta)

            JJT = J.dot(J.T)
            JJT.flat[::JJT.shape[0]+1] += dampening_factors
            theta += J.T.dot(solve_psd(JJT,t-s(theta),overwrite_b=True))

            if np.linalg.norm(s(theta) - t) < tol:
                return theta
        return theta
    return solver

