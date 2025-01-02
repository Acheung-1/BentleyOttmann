# BentleyOttmann

**Problem Statement**

Devise and implement a sweep algorithm to perform the following task: The input is a set S of n line segments in the plane. The algorithm is to determine if the union of these segments is a tree or not (i.e., in the arrangement of the segments, is there a single face, the face at infinity?) An algorithm similar to the Bentley-Ottmann sweep should be possible. To simplify matters, you can use data structures from an appropriate library, or implement trivial, naive data structures. Try to animate the algorithm so it can be viewed and understood by the class. 

**Assumptions**

Assume no 3 points are colinear, no 2 endpoints lie on a horizontal/vertical line.

**My Approach**

Every line segment is represented as a node. I built a custom AVL tree data structure for the Sweep Line Status (SLS) that stores the segments in the left-to-right order that a horizontal line sweeps the planar straight-line graph (PSLG). It determines the ordering by conducting left tests to decide if the node is placed in the left or right child. Each line segment also has an attribute that identifies the set of intersections it has (from top-down order). Every intersection gets a unique label when it pops off the event queue (data structure used is a heap queue). The idea behind this approach is that when there is a face, the topmost vertex will be an intersection, let’s say labeled as intersection A. Line segments 1 and 2 that cross will share their intersection labels (take the union of the intersection label set and give to both segments). As the lines form more intersections (forming the other vertexes) this unique identifier will get passed down to every line segment. At the lowest most vertex, both lines that intersect will both be labeled as having seen intersection A. If two lines that intersect contain the same intersection label, then it has formed a face.

I couldn’t get the AVL tree to work perfectly, when there are swaps and deletions, the nodes ordering might get messed up and sometimes I can’t find the specific node in O(logn) time, so I cheated and made it look through all the nodes in O(n) time only if this is the case. The number of times it does this is printed in the terminal for reference. The number of times it must do a linear scan to find a node is significantly smaller than the number of times it can find it going down the tree.

The code runs in O((n+k)logn) where k is the number of intersections. At any point in time, the event queue only contains one intersection (the top-most intersection that has been currently found that has not been swept yet).
How to run the code:

Can run as a script in the terminal: Python BentleyOttmannSweep.py
Or in VS code.

Another screen will pop up. Draw line segments in the top-left quadrant using point clicks. Two clicks identify the start and end of a line segment. It is advisable to space out intersections and endpoints of segments so the current sweep line status could be visible to the right. When ready, press the start button. 
 
A text file, “segment_coordinates.txt”, will be produced. This contains the number of segments and the coordinates.
 
A matlab plot of the planar graph “segments.png” that was just drawn will be saved for visibility.
 
You can press the start button or spacebar to see the sweep line advance. In the top-right quadrant, the current sweep line status will be printed, the coordinates if it is an intersection, and if a face is detected. Once it reaches the end of the event queue, the number of intersections and the number of faces will be printed out in the bottom-right quadrant. Press exit to close the window.
 

You can change the following in the code:

def main(imp=False, run=False, file="segment_coordinates.txt"):
    IMPORT = imp
    RUNALL = run
    file = file

imp:
–	False: You want to draw the segments
–	True: You want to import a text file of coordinates (must be in correct format)
run:
–	False: Click start or spacebar to advance to the next change in SLS
–	True: Run through entire SLS automatically
file
–	name of the text file

