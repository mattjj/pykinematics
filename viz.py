from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.artist import Artist

import ik
from util import flatten1

# TODO make this an FK thing, not a tree node thing
class JointTreeNodeViz(ik.JointTreeNode):
    def lineplot_xys(self):
        def interleave(item,lst):
            return [v for l in lst for v in (l,item)]

        n = self.E.ndim
        return [np.zeros(n)] + interleave(np.zeros(n),map(self.E.apply,self.effectors)) \
                + flatten1(interleave([np.zeros(n)], [map(self.E.apply,c.lineplot_xys()) for c in self.children]))

class InteractiveIK(object):
    epsilon = 5.

    def __init__(self, fig, ax, tree):
        self.ax = ax
        self.canvas = canvas = fig.canvas

        self.solver = ik.construct_solver(ik.JointTreeFK(tree),dampening_factors=1.,tol=1e-2,maxiter=100)
        self.effectors = np.array(tree.get_effectors())
        self.line = Line2D(*zip(*tree.lineplot_xys()), marker='o', markerfacecolor='b',animated=True)
        self.ax.add_line(self.line)

        self._ind = None # the effector being dragged

        self.line.add_callback(self.line_changed)
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

    def line_changed(self, line):
        vis = self.line.get_visible()
        Artist.update_from(self.line, line)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.line)
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

        self.effectors[self._ind] = event.xdata, event.ydata
        self.tree.set(self.solver(self.effectors.ravel(), self.tree.coordinates))
        self.line.set_data(*zip(*self.tree.lineplot_xys()))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def _get_ind_under_point(self, event):
        xyt = self.line.get_transform().transform(self.effectors)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt-event.x)**2 + (yt-event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind]>=self.epsilon:
            ind = None

        return ind

if __name__ == '__main__':
    fig = plt.figure()
    ax = plt.subplot(111)

    ### 3 joints
    a = JointTreeNodeViz(E=ik.RotorJoint2D(1.5),
            children=[JointTreeNodeViz(E=ik.RotorJoint2D(1.),
                children=[JointTreeNodeViz(E=ik.RotorJoint2D(1.),effectors=[np.zeros(2)])])
            ])
    a.set(np.array([np.pi/4,np.pi/4,np.pi/4]))

    ### 2 joints
    # a = JointTreeNodeViz(E=ik.RotorJoint2D(1.5),children=[JointTreeNodeViz(E=ik.RotorJoint2D(1.),effectors=[np.zeros(2)])])
    # a.set(np.array([np.pi/4,np.pi/4]))

    v = InteractiveIK(fig,ax,a)

    ax.set_xlim((-3,3))
    ax.set_ylim((-3,3))
    plt.show()

# TODO show target
