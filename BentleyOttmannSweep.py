import pygame
import sys
import collections
import heapq
import math
import matplotlib.pyplot as plt
import copy

logTime = 0
linearTime = 0

class Node():
    def __init__(self, start_coords=None, end_coords=None, left=None, right=None, parent=None, height=None, left_neighbor=None, right_neighbor=None):
        self.start_coords = start_coords
        self.end_coords = end_coords
        self.height = height

        self.left = left 
        self.right = right 
        self.parent = parent
        
        self.left_neighbor = left_neighbor
        self.right_neighbor = right_neighbor
        self.intersections = set()
        
class AVLTree():
    def __init__(self, segment_list=[]):
        self.rootNode = None
        self.num_segments = 0
        self.height = -1
        
        self.leftMost = Node(start_coords=(-math.inf))
        self.rightMost = Node(start_coords=(math.inf))
        
        self.leftMost.left_neighbor = self.rightMost
        self.leftMost.right_neighbor = self.rightMost

        self.rightMost.left_neighbor = self.leftMost
        self.rightMost.right_neighbor = self.leftMost

        for seg in segment_list:
            self.insert([seg])
        
    def insert(self, segment_list, point=None):
        # print("Inserting")
        self.num_segments += 1

        start_coords, end_coords = segment_list

        x1, y1 = start_coords
        x2, y2 = end_coords

        if point is None:
            point = start_coords
        
        if self.rootNode is None:
            self.rootNode = Node(height=0, start_coords=(x1, y1), end_coords=(x2, y2), left_neighbor=self.leftMost, right_neighbor=self.rightMost)
            self.leftMost.right_neighbor = self.rootNode
            self.rightMost.left_neighbor = self.rootNode
        else:
            cur = self.rootNode
            prev = None

            while cur:
                prev = cur
                global cur_location
                if left(cur_location[(cur.start_coords, cur.end_coords)], cur.end_coords, point):
                    cur = cur.right
                    move = "right"
                else:
                    cur = cur.left
                    move = "left"

            cur = prev

            if move == "right":
                cur.right = Node(height=0, start_coords=(x1, y1), end_coords=(x2, y2), parent=cur, left_neighbor=cur, right_neighbor=cur.right_neighbor)
                l = cur
                r = cur.right_neighbor
                cur = cur.right
            elif move == "left":
                cur.left = Node(height=0, start_coords=(x1, y1), end_coords=(x2, y2), parent=cur, left_neighbor=cur.left_neighbor, right_neighbor=cur)
                l = cur.left_neighbor
                r = cur
                cur = cur.left
            
            l.right_neighbor = cur
            r.left_neighbor = cur

            self.inOrderTraversal(line_segment_dictionary)
            self.checkBalance(cur)
            
        self.height = self.getHeight()

        return

    def delete(self, segment_list, point=None):
        # print("Deleting")
        start_coords, end_coords = segment_list
        global cur_location
        point = cur_location[(start_coords, end_coords)]

        match = self.findElement((start_coords, end_coords), point)

        # num is not found
        if match is None:
            return

        self.num_segments -= 1


        # Setting up in order
        l = match.left_neighbor
        r = match.right_neighbor

        l.right_neighbor = r
        r.left_neighbor = l
        
        # Need to balance the tree
        parent = match.parent
        left_child = match.left
        right_child = match.right
        predecessor = None

        ### If match has no children
        if left_child is None and right_child is None:
            if parent is None:
                self.rootNode = None
            elif parent.left == match:
                parent.left = None
            elif parent.right == match:
                parent.right = None

        ### If match has one child
        elif left_child and right_child is None:
            left_child.parent = parent if parent else None
            if parent is None:
                self.rootNode = left_child
            else:
                if parent.left == match:
                    parent.left = left_child
                else:
                    parent.right = left_child


        elif left_child is None and right_child:
            right_child.parent = parent if parent else None
            if parent is None:
                self.rootNode = right_child
            else:
                if parent.left == match:
                    parent.left = right_child
                else:
                    parent.right = right_child

        ### If match has left and right child
        elif left_child and right_child:
            
            #look for predecessor
            predecessor = match.left
            start = predecessor
            while predecessor.right:
                predecessor = predecessor.right

            # If predecessor has one child (left)
            if predecessor.left:
                if start != predecessor:
                    predecessor.parent.right = predecessor.left
                    predecessor.left.parent = predecessor.parent
                else:
                    predecessor.parent.left = predecessor.left
                    predecessor.left.parent = predecessor.parent
            else:
                # If predecessor has no child
                if start != predecessor:
                    predecessor.parent.right = None
                else:
                    predecessor.parent.left = None

            predecessor.parent = parent
            predecessor.left = match.left if match.left != predecessor else None
            predecessor.right = match.right

            # Predecessor takes place of match
            if parent:
                if parent.left == match:
                    parent.left = predecessor
                else:
                    parent.right = predecessor
            else:
                self.rootNode = predecessor
            
            if predecessor.left:
                predecessor.left.parent = predecessor

            if predecessor.right:
                predecessor.right.parent = predecessor
            
        self.checkBalance(predecessor if predecessor else match.parent)
        del match
        self.height = self.getHeight()
        
        return
    
    def swap(self, node1_coords, node2_coords):
        # print("Swapping")
        start1, end1 = node1_coords
        start2, end2 = node2_coords

        global cur_location
        node1 = self.findElement(node1_coords, cur_location[(start1, end1)])
            
        node2 = self.findElement(node2_coords, cur_location[(start2, end2)])

        t_start_coords = node1.start_coords
        t_end_coords = node1.end_coords
        t_intersections = node1.intersections

        node1.start_coords = node2.start_coords
        node1.end_coords = node2.end_coords
        node1.intersections = node2.intersections
        
        node2.start_coords = t_start_coords
        node2.end_coords = t_end_coords
        node2.intersections = t_intersections

        # PREVIOUS CHILDREN
        if node1.left and node1.left == node1:
            node1.left = node2
        if node1.right and node1.right == node1:
            node1.right = node2
        if node1.parent and node1.parent == node1:
            node1.parent = node2
        
        
        if node2.left and node2.left == node2:
            node2.left = node1
        if node2.right and node2.right == node2:
            node2.right = node1
        if node2.parent and node2.parent == node2:
            node2.parent = node1

        return node1, node2

    def findElement(self, segment, point):

        start_coords, end_coords = segment
        global cur_location

        cur = self.rootNode
        while cur:
            if cur.start_coords == start_coords and cur.end_coords == end_coords:
                # print("O(logn)")
                global logTime 
                logTime += 1
                return cur
            elif left(cur_location[(cur.start_coords, cur.end_coords)], cur.end_coords, point):
                cur = cur.right
            else:
                cur = cur.left
        
        start = self.leftMost.right_neighbor
        while start != self.rightMost:
            if start.start_coords == start_coords and start.end_coords == end_coords:
                # print("O(n)")
                global linearTime 
                linearTime += 1
                return start
            start = start.right_neighbor

        return None
    
    def checkBalance(self, node):

        while node:
            node.height = max(node.left.height if node.left else -1, node.right.height if node.right else -1) + 1

            if self.requireBalancing(node):
                if (node.left.height if node.left else -1) > (node.right.height if node.right else -1):
                    if (node.left.left.height if node.left.left else -1) > (node.left.right.height if node.left.right else -1):
                        self.rightRotate(node.parent, node, node.left, node.right)
                    else:
                        node = node.left
                        self.leftRotate(node.parent, node, node.left, node.right)
                        node = node.parent.parent
                        self.rightRotate(node.parent, node, node.left, node.right)

                else:
                    if (node.right.right.height if node.right.right else -1) > (node.right.left.height if node.right.left else -1):
                        self.leftRotate(node.parent, node, node.left, node.right)
                    else:
                        node = node.right
                        self.rightRotate(node.parent, node, node.left, node.right)
                        node = node.parent.parent
                        self.leftRotate(node.parent, node, node.left, node.right)

                node = node.parent
            if node == None:
                break
            node = node.parent
        
        return
    
    def leftRotate(self, cur_parent, cur, left, right):
        # definitely have cur and right
        #      parent
        #        |
        #       cur
        #       /  \
        #    left right 
        #             \
        #           right.right
        # print(line_segment_dictionary[cur.start_coords, cur.end_coords])
        if cur_parent is None:
            self.rootNode = right
            right.parent = None
        else:
            if cur_parent.left == cur:
                cur_parent.left = right
            else:
                cur_parent.right = right
            right.parent = cur_parent

        if right.left is not None:
            right.left.parent = cur
        cur.right = right.left
        right.left = cur
        cur.parent = right
        
        cur.height = max(cur.left.height if cur.left else -1, cur.right.height if cur.right else -1) + 1
        right.height = max(right.left.height if right.left else -1, right.right.height if right.right else -1) + 1
        if cur_parent:
            cur_parent.height = max(cur_parent.left.height if cur_parent.left else -1, cur_parent.right.height if cur_parent.right else -1) + 1

        return
    
    def rightRotate(self, cur_parent, cur, left, right):
        # definitely have cur and left 
        #      parent
        #        |
        #       cur
        #       /  \
        #    left right 
        #     /    
        # left.left

        #      parent
        #        |
        #       left
        #       /  \
        # left.left cur 
        #           /     \
        #       left.right  right

        if cur_parent is None:
            self.rootNode = left
            left.parent = None
        else:
            if cur_parent.left == cur:
                cur_parent.left = left
            else:
                cur_parent.right = left
            left.parent = cur_parent

        if left.right is not None:
            left.right.parent  = cur
        cur.left = left.right
        left.right = cur
        cur.parent = left
        
        cur.height = max(cur.left.height if cur.left else -1, cur.right.height if cur.right else -1) + 1
        left.height = max(left.left.height if left.left else -1, left.right.height if left.right else -1) + 1
        if cur_parent:
            cur_parent.height = max(cur_parent.left.height if cur_parent.left else -1, cur_parent.right.height if cur_parent.right else -1) + 1

        return
    
    def requireBalancing(self, node):
        left = node.left.height if node.left else -1
        right = node.right.height if node.right else -1
        
        return abs(left-right) >= 2
    
    def getHeight(self):
        if self.rootNode:
            return self.rootNode.height
        else:
            -1
    
    def orderByLayer(self):

        q = collections.deque()
        q.append(self.rootNode)
        tree = []
        
        i = 0
        while q:
            n = len(q)
            layer = []
            for _ in range(n):
                cur = q.popleft()
                if type(cur) == Node:
                    q.append(cur.left)
                    q.append(cur.right)
                else:
                    q.append(None)
                    q.append(None)
                
                layer.append(cur if cur else None)


            tree.append(layer)
            if all(x is None for x in q):
                break
    
        for i in range(len(tree)):
            group = []
            for j in range(len(tree[i])):
                node = tree[i][j]
                group.append(f"{node.val}" if node else "None")
                # group.append(f"{node.val} with height {node.height} and parent {node.parent.val if node.parent else None}" if node else "None")
            print(f"Layer {i} has "+ ", ".join(group))
        
        return
    
    def inOrderTraversal(self, segment_dictionary=None):

        # print("\norder by tree")
        # stack = []
        # cur = self.rootNode
        # traversal = []

        # while cur or stack:
        #     while cur:
        #         stack.append(cur)
        #         cur = cur.left
        #     cur = stack.pop()
        #     traversal.append(cur)
        #     cur = cur.right

        # for i in range(len(traversal)):
        #     print(f"""Segment {i}: {traversal[i].start_coords} to {traversal[i].end_coords}: 
        #         parent {traversal[i].parent.start_coords if traversal[i].parent else traversal[i].parent} to {traversal[i].parent.end_coords if traversal[i].parent else traversal[i].parent}, 
        #         left child {traversal[i].left.start_coords if traversal[i].left else traversal[i].left} to {traversal[i].left.end_coords if traversal[i].left else traversal[i].left}, 
        #         right child {traversal[i].right.start_coords if traversal[i].right else traversal[i].right} to {traversal[i].right.end_coords if traversal[i].right else traversal[i].right},
        #         left neighbor {traversal[i].left_neighbor.start_coords if traversal[i].left_neighbor else traversal[i].left_neighbor} to {traversal[i].left_neighbor.end_coords if traversal[i].left_neighbor else traversal[i].left_neighbor}
        #         right neighbor {traversal[i].right_neighbor.start_coords if traversal[i].right_neighbor else traversal[i].right_neighbor} to {traversal[i].right_neighbor.end_coords if traversal[i].right_neighbor else traversal[i].right_neighbor}
        #         height {traversal[i].height}""")
            
        # print("\nFROM LEFT TO RIGHT")
            
        cur = self.leftMost
        traversal = []

        while cur.right_neighbor != self.rightMost:
            cur = cur.right_neighbor
            traversal.append(cur)

        # for i in range(len(traversal)):
        #     # print(f"""Segment {i}: {traversal[i].start_coords} to {traversal[i].end_coords}: 
        #     #     parent {traversal[i].parent.start_coords if traversal[i].parent else traversal[i].parent} to {traversal[i].parent.end_coords if traversal[i].parent else traversal[i].parent}, 
        #     #     left child {traversal[i].left.start_coords if traversal[i].left else traversal[i].left} to {traversal[i].left.end_coords if traversal[i].left else traversal[i].left}, 
        #     #     right child {traversal[i].right.start_coords if traversal[i].right else traversal[i].right} to {traversal[i].right.end_coords if traversal[i].right else traversal[i].right},
        #     #     left neighbor {traversal[i].left_neighbor.start_coords if traversal[i].left_neighbor else traversal[i].left_neighbor} to {traversal[i].left_neighbor.end_coords if traversal[i].left_neighbor else traversal[i].left_neighbor}
        #     #     right neighbor {traversal[i].right_neighbor.start_coords if traversal[i].right_neighbor else traversal[i].right_neighbor} to {traversal[i].right_neighbor.end_coords if traversal[i].right_neighbor else traversal[i].right_neighbor}
        #     #     height {traversal[i].height}""")
        #     print(f"""Segment: {segment_dictionary[(traversal[i].start_coords, traversal[i].end_coords)] } : 
        #         parent {segment_dictionary[traversal[i].parent.start_coords, traversal[i].parent.end_coords] if traversal[i].parent else traversal[i].parent} 
        #         left child {segment_dictionary[traversal[i].left.start_coords, traversal[i].left.end_coords] if traversal[i].left else traversal[i].left}
        #         right child {segment_dictionary[traversal[i].right.start_coords, traversal[i].right.end_coords] if traversal[i].right else traversal[i].right} 
        #         left neighbor {segment_dictionary[traversal[i].left_neighbor.start_coords, traversal[i].left_neighbor.end_coords]  if traversal[i].left_neighbor else traversal[i].left_neighbor} 
        #         right neighbor {segment_dictionary[traversal[i].right_neighbor.start_coords, traversal[i].right_neighbor.end_coords] if traversal[i].right_neighbor else traversal[i].right_neighbor}
        #         height {traversal[i].height}""")
        
        if segment_dictionary:
            current_SLS = ""
            for i in range(len(traversal)):
                current_SLS += (segment_dictionary[((traversal[i].start_coords),(traversal[i].end_coords))] + " ")
        
        return current_SLS
    
def area2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

def left(a, b, c):
    return area2(a, b, c) > 0

def colinear(a, b, c):
    return area2(a, b, c) == 0

def xor(x, y):
    if x == True and y == False or x == False and y == True:
        return True
    else:
        return False

def intersect(a, b, c, d):
    return xor( left(a, b, c), left(a, b, d)) and xor( left(c, d, a), left(c, d, b))

def findIntersectionPoint(a, b, c, d):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d

    # Parametric equations of the lines
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    # Denominator for solving the parametric equations
    denom = dx1 * dy2 - dy1 * dx2

    # Parameters for the parametric equations
    t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom
    t2 = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / denom

    intersection_x = x1 + dx1 * t1
    intersection_y = y1 + dy1 * t1

    return round(intersection_x, 10), round(intersection_y, 10)

def save_to_file(segments, fileName, width, height):
    with open(fileName, "w") as file:

        file.write(f"{len(segments)}\n")
        for segment in segments:
            x = 0
            for point in segment:
                if x == 0:
                    file.write(f"{point[0]}, {height - point[1]} with ")
                    x += 1
                else:
                    file.write(f"{point[0]}, {height - point[1]}\n")
                    x = 0

    return

def draw_button(surface, font, color, rect, text, text_color):
    pygame.draw.rect(surface, color, rect)
    font_surface = font.render(text, True, text_color)
    text_rect = font_surface.get_rect(center=rect.center)
    surface.blit(font_surface, text_rect)
    
    return

def is_inside(pos, rect):
    x, y = pos
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh
    
def main(imp=False, run=False, file="segment_coordinates.txt"):
    IMPORT = imp
    RUNALL = run
    file = file

    ##### PYGAME GUI SCREEN
    pygame.init()
    WIDTH, HEIGHT = 1200, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sweep Algorithm Visualization")

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GREY = (127, 127, 127)
    PINK = (255, 0, 255)
    CYAN = (0, 255, 255)

    colors = [BLACK, RED, GREEN, BLUE, GREY, PINK, CYAN]

    font = pygame.font.SysFont(None, 24)

    # Mouse click coordinates
    click_coords = []
    line_segments = []

    DIVIDER = int(WIDTH * 0.6)

    # Button properties
    start_button_rect = pygame.Rect(50, HEIGHT - 75, 100, 40)
    exit_button_rect = pygame.Rect(DIVIDER - 150, HEIGHT - 75, 100, 40) 
    start_button_color = (0, 128, 0)
    start_button_text = "Start"
    start_button_text_color = WHITE
    exit_button_color = RED
    exit_button_text = "Exit"
    exit_button_text_color = WHITE

    # Run
    running = True
    continue_pressed = False
    no_more_continue = True
    drawing_lines = False
    completed = False
    printed = False
    initialize = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if is_inside(event.pos, start_button_rect):
                    continue_pressed = True 
                elif is_inside(event.pos, exit_button_rect):
                    running = False
                else:
                    click_coords.append(event.pos)
                    if len(click_coords) == 2:
                        line_segments.append(click_coords.copy())
                        click_coords.clear()

        if not initialize:
            screen.fill(WHITE)
            pygame.draw.line(screen, BLACK, (DIVIDER, 0), (DIVIDER, HEIGHT), 5)
            pygame.draw.line(screen, BLACK, (0, HEIGHT - 100), (WIDTH, HEIGHT - 100), 5)
            initialize = True

        if not continue_pressed:
            i = 0
            for segment in line_segments:
                pygame.draw.line(screen, colors[i], segment[0], segment[1], 2)
                i += 1
                if i == len(colors):
                    i = 0

        draw_button(screen, font, start_button_color, start_button_rect, start_button_text, start_button_text_color)
        draw_button(screen, font, exit_button_color, exit_button_rect, exit_button_text, exit_button_text_color)

        pygame.display.flip()

        if continue_pressed and no_more_continue:
            ##### SAVE COORDINATES AND FORMAT
            if IMPORT == False:
                save_to_file(line_segments, file, WIDTH, HEIGHT)

            with open(file, "r") as readfile:
                a = readfile.read()       
            lines = a.split("\n")

            line_segments = []

            for line in lines[1:-1]:
                a, b = line.split(" with ")
                ax, ay = a.split(", ")
                bx, by = b.split(", ")

                ax = int(ax)
                ay = int(ay)
                bx = int(bx)
                by = int(by)

                # Add line segments in order from largest y value, if they match then order by smallest x value
                if ay < by or ay == by and ax > by:
                    line_segments.append([(bx, by), (ax, ay)])
                else:
                    line_segments.append([(ax, ay), (bx, by)])
            
            #  Draw segments
            if IMPORT == True:
                screen.fill(WHITE)
                pygame.draw.line(screen, BLACK, (DIVIDER, 0), (DIVIDER, HEIGHT), 5)
                pygame.draw.line(screen, BLACK, (0, HEIGHT - 100), (WIDTH, HEIGHT - 100), 5)
                draw_button(screen, font, start_button_color, start_button_rect, start_button_text, start_button_text_color)
                draw_button(screen, font, exit_button_color, exit_button_rect, exit_button_text, exit_button_text_color)

                i = 0
                for segment in line_segments:
                    pygame.draw.line(screen, colors[i], (segment[0][0], HEIGHT - segment[0][1]), (segment[1][0], HEIGHT - segment[1][1]), 2)
                    i += 1
                    if i == len(colors):
                        i = 0
            

            line_dictionary = {}
            global cur_location
            cur_location = {}
            sorted_segments = sorted(line_segments, key=lambda x: -x[0][1])

            for start, end in sorted_segments:
                line_dictionary[start] = end
                cur_location[(start, end)] = start


            points = []

            for key, value in line_dictionary.items():
                points.append([key, value, "start"])
                points.append([value, key, "end"])

            event_queue = [(-x[0][1], x[0][0], -x[1][1], x[1][0], x[2]) for x in points]
            
            heapq.heapify(event_queue)
            
            intercept_dictionary = {}

            sweep_line_status = AVLTree()
            intersection_label = 0
            i = 0
            intersect_done = set()
            number_intersections = 0
            number_faces = 0
            drawing_lines = True
            no_more_continue = False


            # SAVE AS MATLAB PLOT FOR VERIFICATION
            global line_segment_dictionary
            line_segment_dictionary = {}
            line_segment_dictionary[((-math.inf), None)] = "leftMost"
            line_segment_dictionary[((math.inf), None)] = "rightMost"

            
            i = 0
            event_queue_sorted = sorted(event_queue, key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

            for y1, x1, y2, x2, point_type in event_queue_sorted:
                if point_type == "start":
                    y1 = -y1  # Flip y-coordinates because matplotlib's origin is at the top-left
                    y2 = -y2
                    plt.plot([x1, x2], [y1, y2], label=f"S{i+1}")
                    plt.scatter([x1, x2], [y1, y2])
                    line_segment_dictionary[((x1, y1), (x2, y2))] = f"S{i+1}"
                    for j, (x, y) in enumerate([(x1, y1), (x2, y2)]):
                        plt.text(x, y, f"({x}, {y})", fontsize=9, ha='right' if j == 0 else 'left')
                    i += 1

                
            if IMPORT == False:
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('Planar Straight-Line Graph')
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
                plt.grid(True)
                plt.savefig("segments.png", bbox_inches='tight')

            for key, value in line_segment_dictionary.items():
                if key == ((-math.inf), None) or key == ((math.inf), None):
                    continue
                start, end = key
                text_surface = font.render(value, True, BLACK)
                screen.blit(text_surface, (start[0] - 10, HEIGHT - start[1] - 20))
            pygame.display.flip()


            event_queue_without_intersections = [(-x[0][1], x[0][0], -x[1][1], x[1][0], x[2]) for x in points]
            heapq.heapify(event_queue_without_intersections)
            current_intersection = None
            intersection_stack = []
            intersection_stack_set = set()
            
        ###
        ### IMPLEMENTING BENTLEY OTTOMANN SWEEP
        ###
        if drawing_lines and not completed:
            print("\nNext event")

            y1, x1, y2, x2, point_type = heapq.heappop(event_queue)
            y1 = -y1
            y2 = -y2

            horiz = y1
            intersect_detected = False
            face_detected = False
            x_inter = None
            y_inter = None


            if point_type == "intercept":
                s1_start, s1_end, s2_start, s2_end = intercept_dictionary[(x1, y1)]
                if (x1, y1) in intersect_done:
                    continue
                else:
                    # print(f"There is an intercept at ({x1}, {y1})")
                    intersect_detected = True
                    intersection_coordinates = (round(x1, 2), round(y1, 2))
                    number_intersections += 1
            
            if point_type != "intercept":
                heapq.heappop(event_queue_without_intersections)
            

            if point_type == "start":

                sweep_line_status.insert([(x1, y1), (x2, y2)])
                cur = sweep_line_status.findElement([(x1, y1), (x2, y2)], cur_location[(x1, y1), (x2, y2)])

                l = cur.left_neighbor
                r = cur.right_neighbor

                if l != sweep_line_status.leftMost and intersect(l.start_coords, l.end_coords, (x1, y1), (x2, y2)):
                    x_inter, y_inter = findIntersectionPoint(l.start_coords, l.end_coords, (x1, y1), (x2, y2))
                    if x_inter and y_inter and (x_inter, y_inter) not in intersection_stack_set:
                        heapq.heappush(intersection_stack, (-y_inter, x_inter))
                        intersection_stack_set.add((x_inter, y_inter))
                        intercept_dictionary[(x_inter, y_inter)] = [l.start_coords, l.end_coords, (x1, y1), (x2, y2)]
                    
                if r != sweep_line_status.rightMost and intersect((x1, y1), (x2, y2), r.start_coords, r.end_coords):
                    x_inter, y_inter = findIntersectionPoint((x1, y1), (x2, y2), r.start_coords, r.end_coords)
                    if x_inter and y_inter and (x_inter, y_inter) not in intersection_stack_set:
                        heapq.heappush(intersection_stack, (-y_inter, x_inter))
                        intersection_stack_set.add((x_inter, y_inter))
                        intercept_dictionary[(x_inter, y_inter)] = [(x1, y1), (x2, y2), r.start_coords, r.end_coords]
                

            elif point_type == "end":

                # For endpoints, coordinates are swapped
                cur = sweep_line_status.findElement([(x2, y2), (x1, y1)], cur_location[(x2, y2), (x1, y1)])

                l = cur.left_neighbor
                r = cur.right_neighbor
                sweep_line_status.delete([(x2, y2), (x1, y1)], (x2, y2))
                if l != sweep_line_status.leftMost and r != sweep_line_status.rightMost and intersect(l.start_coords, l.end_coords, r.start_coords, r.end_coords):
                    x_inter, y_inter = findIntersectionPoint(l.start_coords, l.end_coords, r.start_coords, r.end_coords)
                    if x_inter and y_inter and (x_inter, y_inter) not in intersection_stack_set:
                        heapq.heappush(intersection_stack, (-y_inter, x_inter))
                        intersection_stack_set.add((x_inter, y_inter))
                        intercept_dictionary[(x_inter, y_inter)] = [l.start_coords, l.end_coords, r.start_coords, r.end_coords]
            
            elif point_type == "intercept":

                heapq.heappop(intersection_stack)
                
                intersect_done.add((x1, y1))

                s1_start, s1_end, s2_start, s2_end = intercept_dictionary[(x1, y1)]

                node1, node2 = sweep_line_status.swap([s1_start, s1_end], [s2_start, s2_end])
                
                l = node1.left_neighbor
                r = node2.right_neighbor

                cur_location[(s1_start, s1_end)] = (x1,y1)
                cur_location[(s2_start, s2_end)] = (x1,y1)


                if node1.intersections.intersection(node2.intersections):
                    # print("Face detected.")
                    face_detected = True
                    number_faces += 1

                union_set = node1.intersections | node2.intersections 
                union_set.add(intersection_label)
                node1.intersections = union_set
                node2.intersections = union_set
                
                if l != sweep_line_status.leftMost and intersect(l.start_coords, l.end_coords, node1.start_coords, node1.end_coords):
                    x_inter, y_inter= findIntersectionPoint(l.start_coords, l.end_coords, node1.start_coords, node1.end_coords)
                    if x_inter and y_inter and (x_inter, y_inter) not in intersection_stack_set:
                        heapq.heappush(intersection_stack, (-y_inter, x_inter))
                        intersection_stack_set.add((x_inter, y_inter))
                        intercept_dictionary[(x_inter, y_inter)] = [l.start_coords, l.end_coords, node1.start_coords, node1.end_coords]
                
                if r != sweep_line_status.rightMost and intersect(node2.start_coords, node2.end_coords, r.start_coords, r.end_coords):
                    x_inter, y_inter = findIntersectionPoint(node2.start_coords, node2.end_coords, r.start_coords, r.end_coords)
                    if x_inter and y_inter and (x_inter, y_inter) not in intersection_stack_set:
                        heapq.heappush(intersection_stack, (-y_inter, x_inter))
                        intersection_stack_set.add((x_inter, y_inter))
                        intercept_dictionary[(x_inter, y_inter)] = [node2.start_coords, node2.end_coords, r.start_coords, r.end_coords]
                
                intersection_label += 1
            
            # Endpoint has been considered, cleaning up

            current_intersection = intersection_stack[0] if intersection_stack else None
            
            if current_intersection:
                event_queue = copy.copy(event_queue_without_intersections)
                heapq.heappush(event_queue, (current_intersection[0], current_intersection[1], -math.inf, -math.inf, "intercept"))
            
            pygame.draw.line(screen, BLACK, (0, HEIGHT-horiz), (DIVIDER, HEIGHT-horiz), 2)
            pygame.display.flip()

            ### Printing sweepline status
            text = sweep_line_status.inOrderTraversal(line_segment_dictionary)

            if intersect_detected:
                text += ("   " + str(intersection_coordinates))
                
            if face_detected:
                text += "   Face Detected"

            print(text)

            font_smaller = pygame.font.SysFont(None, 15)
            text_surface = font_smaller.render(text, True, BLACK)
            screen.blit(text_surface, (DIVIDER + 10, HEIGHT - horiz + 10))
            pygame.display.flip()

            if not event_queue:
                completed = True
                continue

            # Wait for user response before continuing
            user_response = False
            if RUNALL:
                user_response = True
            while not user_response:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                    # Spacebar will continue to next event
                        if event.key == pygame.K_SPACE:  
                            user_response = True
                            break  
                    # Or mouse click
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # Start will continue to next event
                        if is_inside(event.pos, start_button_rect):
                            user_response = True
                            break
                        # Exit will exit
                        if is_inside(event.pos, exit_button_rect):
                            user_response = True
                            running = False 
                            break
    
        
        if completed and not printed:
            # print(f"Number of intersections is {number_intersections}")
            # print(f"Number of faces is {number_faces}")
            print(f"logTime: {logTime} vs linearTime: {linearTime}")

            text1 = (f"Number of intersections is {number_intersections}")
            text2 = (f"Number of faces is {number_faces}")

            text_surface1 = font.render(text1, True, BLACK)
            text_surface2 = font.render(text2, True, BLACK)
        
            screen.blit(text_surface1, (DIVIDER + 20, HEIGHT - 80))
            screen.blit(text_surface2, (DIVIDER + 20, HEIGHT - 60))
            pygame.display.flip()
            printed = True
        

    pygame.quit()
    sys.exit()

main()

# if __name__ == "__main__":
#     import sys

#     while len(sys.argv) < 4:
#         sys.argv.append(0)

#     IMPORT = bool(int(sys.argv[1]))
#     RUNALL = bool(int(sys.argv[2]))
#     file = sys.argv[3] if sys.argv[3] != 0 else "segment_coordinates.txt"
#     main(imp=IMPORT, run=RUNALL, file=file)
