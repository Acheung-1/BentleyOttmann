# BentleyOttmann

**Problem Statement**

The input is a set S of n line segments in the plane. The algorithm is to determine if the union of these segments forms a face or a tree.

**Assumptions**

Assume no 3 points are colinear, no 2 endpoints lie on a horizontal/vertical line.

**Background**

The Bentley-Ottmann sweep line algorithm is an efficient method for detecting all intersections among a set of line segments in a plane, operating in O((n+k)logn) time, where n is the number of segments and k is the number of intersections. The algorithm uses a sweep line approach, processing events from top to bottom based on decreasing y-coordinate. It maintains an event queue, implemented as a min-heap, which stores three types of events: segment start points, segment end points, and intersection points. An AVL tree keeps track of the active line segments intersected by the sweep line, ordered from left to right, at any given time. As the sweep line moves downward, segments are inserted into or removed from the BST, and intersections between adjacent segments are identified and processed. When an intersection occurs, the involved segments swap positions in the AVL tree, and potential new intersections are checked efficiently. At any moment, the event queue contains only the highest intersection that has yet to be processed.

**My Approach**

Each line segment is represented as a node within a custom-built AVL tree, designed to manage the Sweep Line Status (SLS). This structure maintains the segments in left-to-right order as a horizontal sweep line progresses through the planar straight-line graph (PSLG). The tree determines the placement of nodes by performing left tests, deciding whether a segment should be inserted as a left or right child.

Additionally, each line segment is assigned an attribute that tracks its set of intersections in top-down order. Intersections are processed using a heap-based event queue, where each intersection receives a unique label upon being dequeued. The core idea behind this approach is that the topmost vertex of a face will always be an intersection, which we can label as intersection A. When two segments (e.g., segment 1 and segment 2) intersect, they share their intersection labels by merging their label sets and passing them to both segments. As more intersections occur along the segments, this label propagates downward. By the time the lowest vertex is reached, both intersecting segments will carry the same label, indicating that they have previously met at an upper intersection. If two intersecting segments already share the same label, a closed face has been successfully detected.

**How to run the code**
Can run as a script in the terminal: Python BentleyOttmannSweep.py

Draw line segments in the top-left quadrant using point clicks. Two clicks identify the start and end of a line segment. It is advisable to space out intersections and endpoints of segments so the current sweep line status could be visible to the right. When ready, press the start button. 
 
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
