#Python 2/3 compatibility
from __future__ import print_function,division,absolute_import
from builtins import input,range
from six import iteritems

from ikdb import *
from ikdb import functionfactory
from klampt import *
import pkg_resources
if pkg_resources.get_distribution('klampt').version >= '0.7':
    NEW_KLAMPT = True
    from klampt.model import ik
    from klampt.io import loader
    from klampt.vis.glrobotprogram import *
    from klampt.vis.glcommon import *
    from klampt import PointPoser,TransformPoser
    from klampt.model import collide
    from klampt.math import se3
    #patch to Klamp't 0.6.X
    class GLWidgetProgram(GLPluginProgram):
        def __init__(self,world,name):
            GLPluginProgram.__init__(self,name)
            self.widgetPlugin = GLWidgetPlugin()
            self.setPlugin(self.widgetPlugin)
            self.widgetMaster = self.widgetPlugin.klamptwidgetmaster
            self.world = world
        def display(self):
            GLPluginProgram.display(self)
            self.world.drawGL()
else:
    NEW_KLAMPT = False
    from klampt import ik,loader
    from klampt.glrobotprogram import *
    from klampt import PointPoser,TransformPoser
    from klampt import robotcollide as collide
    from klampt import se3
import sys
import traceback
import numpy as np


#preload
from sklearn.neighbors import NearestNeighbors,BallTree

class IKDBVisualTester(GLWidgetProgram):
    def __init__(self,visWorld,planningWorld,name="IK Database visual tester"):
        GLWidgetProgram.__init__(self,visWorld,name)
        self.planningWorld = planningWorld
        self.collider = collide.WorldCollider(visWorld)
        self.ikdb = ManagedIKDatabase(planningWorld.robot(0))
        self.ikWidgets = []
        self.ikIndices = []
        self.ikProblem = IKProblem()
        self.ikProblem.setFeasibilityTest('collisionFree',None)
        qmin,qmax = planningWorld.robot(0).getJointLimits()
        self.ikProblem.setCostFunction('jointRangeCost_dynamic',[qmin,qmax])
        self.drawDb = False
        self.continuous = False
        self.reSolve = False
        self.currentConfig = self.world.robot(0).getConfig()

    def mousefunc(self,button,state,x,y):
        #Put your mouse handler here
        #the current example prints out the list of objects clicked whenever
        #you right click
        GLWidgetProgram.mousefunc(self,button,state,x,y)
        self.reSolve = False
        dragging = False
        if NEW_KLAMPT:
            dragging = self.widgetPlugin.klamptwidgetdragging
        else:
            dragging = self.draggingWidget
        if not dragging and button == 2 and state==0:
            #down
            clicked = self.click_world(x,y)
            if clicked is not None and isinstance(clicked[0],RobotModelLink):
                #make a new widget
                link, wpt = clicked
                lpt = se3.apply(se3.inv(link.getTransform()),wpt)
                self.ikIndices.append(len(self.ikWidgets))
                self.ikWidgets.append(PointPoser())
                self.ikWidgets[-1].set(wpt)
                self.widgetMaster.add(self.ikWidgets[-1])
                self.ikProblem.addObjective(ik.objective(link,local=lpt,world=wpt))
                GLWidgetProgram.mousefunc(self,button,state,x,y)
                self.refresh()
            return
        
    #the dx,dy arguments are needed to be cross-compatible between 0.6.x and 0.7
    def motionfunc(self,x,y,dx=0,dy=0):
        dragging = False
        if NEW_KLAMPT:
            retval = GLWidgetProgram.motionfunc(self,x,y,dx,dy)
            dragging = self.widgetPlugin.klamptwidgetdragging
        else:
            retval = GLWidgetProgram.motionfunc(self,x,y)
            dragging = self.draggingWidget
        if dragging:
            #update all the IK objectives
            for i in range(len(self.ikWidgets)):
                index = self.ikIndices[i]
                if isinstance(self.ikWidgets[i],PointPoser):
                    wptnew = self.ikWidgets[i].get()
                    obj = self.ikProblem.objectives[index]
                    link = obj.link()
                    lpt,wptold = obj.getPosition()
                    obj.setFixedPoint(link,lpt,wptnew)
                    #don't solve now, wait for refresh to process
                    if self.continuous and wptnew != wptold:
                        self.reSolve = True
                elif isinstance(self.ikWidgets[i],TransformPoser):
                    Rnew,tnew = self.ikWidgets[i].get()
                    obj = self.ikProblem.objectives[index]
                    link = obj.link()
                    Rold,told = obj.getTransform()
                    obj.setFixedTransform(link,Rnew,tnew)
                    #don't solve now, wait for refresh to process
                    if self.continuous and (Rnew,tnew) != (Rold,told):
                        self.reSolve = True
        return retval

    def keyboardfunc(self,c,x,y):
        if c=='h':
            print ('HELP:')
            print ('[right-click]: add a new IK constraint')
            print ('[space]: tests the current configuration')
            print ('d: deletes IK constraint')
            print ('t: adds a new rotation-fixed IK constraint')
            print ('f: flushes the current database to disk')
            print ('s: saves the current database to disk')
            print ('b: performs one background step')
            print ('B: starts / stops the background thread')
            print ('v: toggles display of the database')
            print ('c: toggles continuous re-solving of IK constraint its as being moved')
            print ('o: toggles soft / hard IK constraints')
        elif c==' ':
            self.planningWorld.robot(0).setConfig(self.currentConfig)
            soln = self.ikdb.solve(self.ikProblem)
            if soln:
                print ("Solved")
                self.currentConfig = soln
                self.refresh()
            else:
                print ("Failure")
        elif c=='d':
            for i,w in enumerate(self.ikWidgets):
                if w.hasHighlight():
                    print ("Deleting IK widget")
                    #delete it
                    index = self.ikIndices[i]
                    self.widgetMaster.remove(w)
                    del self.ikWidgets[i]
                    del self.ikIndices[i]
                    del self.ikProblem.objectives[index]
                    for j in range(len(self.ikIndices)):
                        self.ikIndices[j] = j
                    self.refresh()
                    break
        elif c=='t':
            clicked = self.click_world(x,y)
            if clicked is not None and isinstance(clicked[0],RobotModelLink):
                #make a new widget
                link, wpt = clicked
                Tlink = link.getTransform()
                self.ikIndices.append(len(self.ikWidgets))
                self.ikWidgets.append(TransformPoser())
                self.ikWidgets[-1].set(*Tlink)
                self.widgetMaster.add(self.ikWidgets[-1])
                self.ikProblem.addObjective(ik.objective(link,R=Tlink[0],t=Tlink[1]))
                self.refresh()
        elif c=='f':
            self.ikdb.flush()
        elif c=='s':
            self.ikdb.save()
        elif c=='b':
            self.ikdb.backgroundStep()
            self.refresh()
        elif c=='B':
            if hasattr(self.ikdb,'thread'):
                self.ikdb.stopBackgroundLoop()
            else:
                self.ikdb.startBackgroundLoop(0)
        elif c=='v':
            self.drawDb = not self.drawDb
        elif c=='c':
            self.continuous = not self.continuous
        elif c == 'o':
            self.ikProblem.setSoftObjectives(not self.ikProblem.softObjectives)

    def display(self):
        if self.reSolve:
            self.planningWorld.robot(0).setConfig(self.currentConfig)
            soln = self.ikdb.solve(self.ikProblem)
            if soln:
                self.currentConfig = soln
            self.reSolve = False

        self.world.robot(0).setConfig(self.currentConfig)
        GLWidgetProgram.display(self)
        glDisable(GL_LIGHTING)
        #draw IK goals
        for obj in self.ikProblem.objectives:
            linkindex = obj.link()
            link =  self.world.robot(0).link(linkindex)
            lp,wpdes = obj.getPosition()
            wp = se3.apply(link.getTransform(),lp)
            glLineWidth(4.0)
            glDisable(GL_LIGHTING)
            glColor3f(0,1,0)
            glBegin(GL_LINES)
            glVertex3f(*wp)
            glVertex3f(*wpdes)
            glEnd()
            glLineWidth(1)
        #draw end positions of solved problems
        if self.drawDb:
            glPointSize(3.0)
            glBegin(GL_POINTS)
            for k,db in iteritems(self.ikdb.databases):
                for i in range(db.numProblems()):
                    try:
                        p = db.getProblem(i)
                    except Exception as e:
                        traceback.print_exc()
                        exit(0)
                    if db.solutions[i] is None:
                        glColor3f(1,0,0)
                    else:
                        glColor3f(0,0,1)
                    for obj in p.objectives:
                        lp,wpdes = obj.getPosition()
                        glVertex3f(*wpdes)
            glColor3f(1,1,0)
            for pjson,soln in self.ikdb.backburner:
                p = IKProblem()
                p.fromJson(pjson)
                for obj in p.objectives:
                    lp,wpdes = obj.getPosition()
                    glVertex3f(*wpdes)
            glEnd()
        return

    def click_world(self,x,y):
        """Helper: returns (obj,pt) where obj is the closest world object
        clicked, and pt is the associated clicked point (in world coordinates).

        If no point is clicked, returns None."""
        #get the viewport ray
        if NEW_KLAMPT:
            (s,d) = self.view.click_ray(x,y)
        else:
            (s,d) = self.click_ray(x,y)

        #run the collision tests
        collided = []
        for g in self.collider.geomList:
            (hit,pt) = g[1].rayCast(s,d)
            if hit:
                dist = vectorops.dot(vectorops.sub(pt,s),d)
                collided.append((dist,g[0]))
        if len(collided)==0:
            return None
        dist,obj = min(collided,key=lambda x:x[0])
        return obj,vectorops.madd(s,d,dist)

def main():
    print ("ikdbtest2.py: This example visually shows the learning process")
    print ("USAGE: ikdbtest2.py [ROBOT OR WORLD FILE]")
    print ("Press h for help.")

    import sys
    import os
    fn = os.path.expanduser("~/Klampt-examples/data/robots/tx90ball.rob")
    if len(sys.argv) > 1:
        fn = sys.argv[1]

    world = WorldModel()
    world.readFile(fn)
    planningWorld = world.copy()

    #for free base robots
    qmin,qmax = world.robot(0).getJointLimits()
    for i,(a,b) in enumerate(zip(qmin,qmax)):
        if not np.isfinite(a):
            print ("Setting finite bound on joint",i)
            qmin[i] = -math.pi
        if not np.isfinite(b):
            print ("Setting finite bound on joint",i)
            qmax[i] = math.pi
    planningWorld.robot(0).setJointLimits(qmin,qmax)

    functionfactory.registerDefaultFunctions()
    functionfactory.registerCollisionFunction(planningWorld)
    functionfactory.registerJointRangeCostFunction(planningWorld.robot(0))    

    tester = IKDBVisualTester(world,planningWorld)
    tester.run()

if __name__ == "__main__":
    main()

