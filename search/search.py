# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from game import Directions
    South = Directions.SOUTH
    West = Directions.WEST
    East = Directions.EAST
    North = Directions.NORTH
    #The problem statement is to reach the goal using Depth First Search
    visited = [] #We maintain a data structure to hold all the visited nodes
    dfs_stack = [] #This stack will keep growing as we go to a particular node with its successors
    begin = problem.getStartState()
    dfs_stack.append([begin,[]])
    #Until the stack is empty we keep visiting every node or stop if we reach the goal state
    while bool(dfs_stack):
        currPointList = dfs_stack.pop() #We traverse to the node that was added most recently
        currPoint = currPointList[0]
        currPath = currPointList[1]
        #If the current state is not visited we check if it is the goal state and if not add its successors into the stack
        if currPoint not in visited:
            visited.append(currPoint)
            #If the goal is reached we return the path traversed until now to reach the goal
            if problem.isGoalState(currPoint):
                return currPath
            else:
                for i in problem.getSuccessors(currPoint):
                    nextPoint = i[0]
                    nextDirection = i[1]
                    #append the stack with next possible node and the path to reach the node
                    dfs_stack.append([nextPoint,currPath + [nextDirection]])
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    South = Directions.SOUTH
    West = Directions.WEST
    East = Directions.EAST
    North = Directions.NORTH
    #The problem statement is to reach the goal using Breadth First Search
    visited = [] #We maintain a data structure to hold all the visited nodes
    bfs_queue = [] #This stack will keep growing as we go to a particular node with its successors
    begin = problem.getStartState()
    bfs_queue.append([begin,[]])
    #Until the stack is empty we keep visiting every node or stop if we reach the goal state
    while bool(bfs_queue):
        currPointList = bfs_queue.pop(0) #We traverse to the node in the order it was added into the stack
        currPoint = currPointList[0]
        currPath = currPointList[1]
        #If the current state is not visited we check if it is the goal state and if not add its successors into the stack
        if currPoint not in visited:
            visited.append(currPoint)
            if problem.isGoalState(currPoint):
                #If the goal is reached we return the path traversed until now to reach the goal
                return currPath
            else:
                for i in problem.getSuccessors(currPoint):
                    nextPoint = i[0]
                    nextDirection = i[1]
                    if not nextPoint in visited:
                        #append the stack with next possible node and the path to reach the node
                        bfs_queue.append([nextPoint,currPath + [nextDirection]])
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    South = Directions.SOUTH
    West = Directions.WEST
    East = Directions.EAST
    North = Directions.NORTH
    #The problem statement is to reach the goal using Uniform Cost Search
    visited = [] #We maintain a data structure to hold all the visited nodes
    ucs_queue = [] #This stack will keep growing as we go to a particular node with its successors
    begin = problem.getStartState()
    ucs_queue.append([begin,[],0])
    #We keep traversing the stack until we reach the goal or the stack gets empty
    while len(ucs_queue):
        #The stack is now sorted to find the node that needs to be traversed next and this is done by picking the node with least cost
        ucs_queue = sorted(ucs_queue, key = lambda x : x[2])#We sort the data structure on distance and take the first node
        currPointList = ucs_queue.pop(0)
        currPoint = currPointList[0]
        currPath = currPointList[1]
        currCost = currPointList[2]
        #Visit all the nodes that are not visited in the stack
        if currPoint not in visited:
            visited.append(currPoint)
            #If the goal is reached return the path to reach this state
            if problem.isGoalState(currPoint):
                return currPath
            else:
                for i in problem.getSuccessors(currPoint):
                    nextPoint = i[0]
                    nextDirection = i[1]
                    nextCost = float(i[2])
                    exist_flag = False
                    for j in ucs_queue:
                        if nextPoint == j[0]:
                            if j[2] > currCost+nextCost:
                                j[2] = currCost+nextCost
                                j[1] = currPath + [nextDirection]
                            exist_flag = True
                    if not exist_flag:
                        #If the current node is not the goal state then calculate the cost and path to reach the successors from current state and add them to the stack
                        if nextPoint not in visited:
                            ucs_queue.append([nextPoint,currPath + [nextDirection],(currCost+nextCost)])
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    South = Directions.SOUTH
    West = Directions.WEST
    East = Directions.EAST
    North = Directions.NORTH
    #The problem statement is to reach the goal using A Star Search
    visited = [] #We maintain a data structure to hold all the visited nodes
    ucs_queue = [] #This stack will keep growing as we go to a particular node with its successors
    begin = problem.getStartState()
    ucs_queue.append([begin,[],0])
    #We keep traversing the stack until we reach the goal or the stack gets empty
    while len(ucs_queue):
        #The stack is now sorted to find the node that needs to be traversed next and this is done by picking the node with least cost
        ucs_queue = sorted(ucs_queue, key = lambda x : (x[2]))#We sort the data structure on distance and take the first node
        currPointList = ucs_queue.pop(0)
        currPoint = currPointList[0]
        currPath = currPointList[1]
        currCost = currPointList[2]
        #Visit all the nodes that are not visited in the stack
        if currPoint not in visited:
            visited.append(currPoint)
            #If the goal is reached return the path to reach this state
            if problem.isGoalState(currPoint):
                return currPath
            else:
                for i in problem.getSuccessors(currPoint):
                    nextPoint = i[0]
                    nextDirection = i[1]
                    nextCost = float(i[2])
                    exist_flag = False
                    completeCost = problem.getCostOfActions(currPath+[nextDirection])
                    for j in ucs_queue:
                        if nextPoint == j[0]:
                            if j[2] > completeCost+heuristic(j[0],problem):
                                j[2] = (completeCost + heuristic(j[0],problem))
                                j[1] = currPath + [nextDirection]
                            exist_flag = True
                    if not exist_flag:
                        #If the current node is not the goal state then calculate the cost using a heuristic and path to reach the successors from current state and add them to the stack
                        if nextPoint not in visited:
                            ucs_queue.append([nextPoint,currPath + [nextDirection],(completeCost+heuristic(nextPoint,problem))])
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
