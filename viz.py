from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.patches import Circle

import ik

class JointTreeFKViz(ik.JointTreeFK):
    def lineplot_xys(self):
        return np.asarray(self._lineplot_xys(self.root))[:,:-1]

    def _lineplot_xys(self,node):
        n = node.E.ndim
        origin = np.zeros(n+1)
        origin[-1] = 1
        return [origin] + self._interleave(origin,map(node.E.apply,(e for i,e in node.effectors))) \
                + self._flatten1(self._interleave([origin],
                    [map(node.E.apply,self._lineplot_xys(c)) for c in node.children]))

    def _interleave(self,item,lst):
        return [v for l in lst for v in (l,item)]

    def _flatten1(self,lol):
        return reduce(list.__iadd__,lol,[])


class BigchartJointTreeFKViz(JointTreeFKViz,ik.BigchartJointTreeFK):
    pass

class InteractiveIK(object):
    epsilon = 5. # pixel units

    def __init__(self, fig, ax, tree, initial_coordinates, solver):
        self.ax = ax
        self.canvas = canvas = fig.canvas

        self.tree = tree
        self.solver = solver
        self.prev_coordinates = initial_coordinates

        self.targets = self.tree(initial_coordinates)
        self.circles = [Circle(t, radius=0.1, facecolor='r', alpha=0.5, animated=True) for t in self.targets]
        for c in self.circles:
            self.ax.add_patch(c)

        self.line = Line2D(*zip(*self.tree.lineplot_xys()), marker='o', markerfacecolor='b', animated=True)
        self.ax.add_line(self.line)

        self._ind = None # the target being dragged

        self.line.add_callback(self.line_changed)
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

    def line_changed(self, line):
        vis = self.line.get_visible()
        Artist.update_from(self.line, line)
        self.line.set_visible(vis)

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.line)
        for c in self.circles:
            self.ax.draw_artist(c)
        self.canvas.blit(self.ax.bbox)

    def button_press_callback(self, event):
        if event.inaxes==None: return
        if event.button != 1: return
        self._ind = self._get_ind_under_point(event)

    def button_release_callback(self, event):
        if event.button != 1: return
        self._ind = None

    def motion_notify_callback(self, event):
        if self._ind is None: return
        if event.inaxes is None: return
        if event.button != 1: return
        x,y = event.xdata, event.ydata

        self.targets[self._ind] = x,y
        self.circles[self._ind].center = x,y

        self.prev_coordinates = self.solver(self.targets, self.prev_coordinates)
        # self.tree(self.prev_coordinates)
        self.line.set_data(*zip(*self.tree.lineplot_xys()))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.line)
        for c in self.circles:
            self.ax.draw_artist(c)
        self.canvas.blit(self.ax.bbox)

    def _get_ind_under_point(self, event):
        # TODO can get the circle index instead
        xyt = self.line.get_transform().transform(self.targets)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt-event.x)**2 + (yt-event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind]>=self.epsilon:
            ind = None

        return ind

##############
#  examples  #
##############

def chain_example():
    fig = plt.figure()
    ax = plt.subplot(111)

    treeroot = ik.JointTreeNode(E=ik.RotorJoint2D(1.5),
            children=[ik.JointTreeNode(E=ik.RotorJoint2D(1.),effectors=[np.zeros(2)],
                children=[ik.JointTreeNode(E=ik.RotorJoint2D(1.),effectors=[np.zeros(2)])])
            ])
    tree = JointTreeFKViz(treeroot)
    solver = ik.construct_solver(tree,dampening_factors=1.,tol=1e-2,maxiter=30,errorlimits=(-2,2))
    v = InteractiveIK(fig,ax,tree,np.array([np.pi/4,np.pi/4,np.pi/4]),solver)

    ax.set_xlim((-4,4))
    ax.set_ylim((-4,4))

    return v

def tree_example():
    fig = plt.figure()
    ax = plt.subplot(111)

    # NOTE: shape and lengths match tree constructed below
    temp = np.zeros((4,3,3))
    temp[:,-1,-1] = 1.
    lengths = np.array([1.5,1.,1.,1.])
    def bigchart(thetas,out):
        c, s = np.cos(thetas), np.sin(thetas)
        out[:,0,0] = out[:,1,1] = c
        out[:,0,1] = -s
        out[:,1,0] = s
        out[:,0,-1] = c*lengths
        out[:,1,-1] = s*lengths

    treeroot = ik.JointTreeNode(E=ik.RotorJoint2D(1.5),
            children=[ik.JointTreeNode(E=ik.RotorJoint2D(1.),
                children=[
                    ik.JointTreeNode(E=ik.RotorJoint2D(1.),effectors=[np.zeros(2)]),
                    ik.JointTreeNode(E=ik.RotorJoint2D(1.),effectors=[np.zeros(2)])
                    ])
            ])
    tree = BigchartJointTreeFKViz(treeroot,bigchart,temp)
    solver = ik.construct_solver(tree,dampening_factors=1.,tol=1e-2,maxiter=30,errorlimits=(-2,2))
    v = InteractiveIK(fig,ax,tree, np.array([np.pi/4,np.pi/4,np.pi/4,0.]),solver)

    ax.set_xlim((-4,4))
    ax.set_ylim((-4,4))

    return v

def constrained_chain_example():
    fig = plt.figure()
    ax = plt.subplot(111)

    # NOTE: shape and lengths match tree constructed below
    temp = np.zeros((5,3,3))
    temp[:,-1,-1] = 1.
    lengths = np.array([1.,1.,1.,1.,1.])
    def bigchart(thetas,out):
        c, s = np.cos(thetas), np.sin(thetas)
        out[:,0,0] = out[:,1,1] = c
        out[:,0,1] = -s
        out[:,1,0] = s
        out[:,0,-1] = c*lengths
        out[:,1,-1] = s*lengths

    treeroot = ik.JointTreeNode(E=ik.RotorJoint2D(1.),
            children=[ik.JointTreeNode(E=ik.RotorJoint2D(1.),
                children=[
                    ik.JointTreeNode(E=ik.RotorJoint2D(1.),children=[
                        ik.JointTreeNode(E=ik.RotorJoint2D(1.),children=[
                            ik.JointTreeNode(E=ik.RotorJoint2D(1.),effectors=[np.zeros(2)])])
                        ])
                    ])
            ])
    tree = BigchartJointTreeFKViz(treeroot,bigchart,temp)

    # quadratic potentials
    biaspotentials = np.array([0.25,0.1,0.01,0,0])
    defaulttheta = np.zeros(5)
    def biasfunc(theta):
        return (defaulttheta-theta) * biaspotentials

    solver = ik.construct_biased_solver(
            tree,
            dampening_factors=1.,
            biasfunc=biasfunc,
            tol=1e-2,maxiter=50,
            jointmins=-np.pi/3,jointmaxes=np.pi/3,
            errorlimits=(-2,2))
    v = InteractiveIK(fig,ax,tree, np.array([0.,0.,0.,0.,0.]),solver)

    ax.set_xlim((-6,6))
    ax.set_ylim((-6,6))

    return v


if __name__ == '__main__':
    # v = constrained_chain_example()
    v = tree_example()
    plt.show()

