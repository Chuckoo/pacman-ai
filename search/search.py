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

    prev_node = {} #this dictionary will store the previous node in the path for every node. for example if state2 is reached from state1 by going south, then prev_node[state2] = [state1, South]
    res = [] #final list of directions to reach the goal.
    visited = set() # keep track of visited nodes.
    stack = util.Stack() # using stack as fringe for DFS.

    start_state = problem.getStartState()
    stack.push(start_state)
    goal_state = None

    # pop from the stack until it is empty.
    while not stack.isEmpty():
        cur = stack.pop()
        visited.add(cur)
        if problem.isGoalState(cur):
            goal_state = cur  
            break #stop searching if goal is found.
        for nei in problem.getSuccessors(cur):
            # explore neighbors of cur if they are not yet visited.
            if nei[0] not in visited:
                prev_node[nei[0]] = [cur,nei[1]] # store that this neighbor is reached from cur.
                stack.push(nei[0])

    temp = goal_state
    # backtrack from goal state to get the list of nodes visited to reach the goal.
    while temp != start_state:
        res.insert(0,prev_node[temp][1])
        temp = prev_node[temp][0]
    
    return res

    util.raiseNotDefined()

def breadthFirstSearch(problem):

    prev_node = {} #this dictionary will store the previous node in the path for every node. 
    res = [] #final list of directions to reach the goal.
    visited = set() # keep track of visited nodes.
    queue = util.Queue() # using queue as fringe for BFS.

    start_state = problem.getStartState()
    queue.push(start_state)
    visited.add(start_state)
    goal_state = None

    # pop from the queue until it is empty.
    while not queue.isEmpty():
        cur = queue.pop()
        if problem.isGoalState(cur):
            goal_state = cur
            break # stop searching if goal is found.
        for nei in problem.getSuccessors(cur):
            # explore neighbors of cur if they are not yet visited.
            if nei[0] not in visited:
                visited.add(nei[0])
                prev_node[nei[0]] = [cur, nei[1]] # store that this neighbor is reached from cur.
                queue.push(nei[0])
    
    temp = goal_state
    # backtrack from goal state to get the list of nodes visited to reach the goal.
    while temp != start_state:
        res.insert(0,prev_node[temp][1])
        temp = prev_node[temp][0]

    return res

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    prev_node = {} # this dictionary will store the previous node in the path for every node. 
    res = [] # final list of directions to reach the goal.
    visited = set() #keep track of visited nodes.
    queue = util.PriorityQueue() # using priority queue as fringe for UCS.

    start_state = problem.getStartState()
    queue.push(start_state,0) # push start_state, cost
    visited.add(start_state)
    goal_state = None
    costs = {start_state : 0} # dictionary to store the least cost to reach every node.

    # pop from the priority queue until it is empty.
    while not queue.isEmpty():
        cur_node = queue.pop()
        cur_cost = costs[cur_node]
        visited.add(cur_node)
        if problem.isGoalState(cur_node):
            goal_state = cur_node
            break # stop searching if goal is found.
        for nei in problem.getSuccessors(cur_node):
            # explore neighbors of cur if they are not yet visited.
            if nei[0] not in visited:
                if nei[0] not in prev_node.keys(): #if visiting for the first time, simply add this to prev_node.
                    prev_node[nei[0]] = [cur_node, nei[1]]
                else: # else, update the prev_node of this neighbor if the current cost to reach this is than the existing ones.
                    if costs[nei[0]] > cur_cost + nei[2]:
                        prev_node[nei[0]] = [cur_node, nei[1]]
                #update queue and costs for this neighbor.
                queue.update(nei[0], cur_cost + nei[2])
                costs.update({nei[0] : cur_cost + nei[2]})
                
                
    temp = goal_state
    # backtrack from goal state to get the list of nodes visited to reach the goal.
    while temp != start_state:
        res.insert(0, prev_node[temp][1])
        temp = prev_node[temp][0]

    return res

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
    import util

    """
    |---------------------------|
    | Node  |direction| cost    |
    | coords| to Node |(prioity)|
    |---------------------------|
    """
    pq = util.PriorityQueue() #using this to keep track of lowest c + h node
    visited = [] #keeps track of nodes already visited 
    solution = []
    node_path_map = {}

    # if dot is there at the start only
    if problem.isGoalState(problem.getStartState()):
        return []
    
    pq.push([problem.getStartState(),[],0],0)

    while not pq.isEmpty(): #until there is no more nodes to be searched
        curr = pq.pop()
        solution = curr[1]
        d = curr[2]
        curr = curr[0]
        if problem.isGoalState(curr): #exit condition of solution being found
            print "Total cost", d
            print "Solution found", solution
            return solution
        if curr not in visited:
            visited.append(curr) 
            adjNodes = problem.getSuccessors(curr) # fn returns [(s,a,s),(),(),()] where (s,a,s) is one adjacent node
            for successor,action,stepCost in adjNodes:
                directions = solution + [action]
                c = problem.getCostOfActions(directions) + heuristic(successor,problem) #cost = g + h
                if successor not in visited:
                    pq.push([successor,directions,c],c)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

