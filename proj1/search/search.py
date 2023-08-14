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

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
   
    stack = util.Stack()
    paths = util.Stack()
    visited = set()
    node = problem.getStartState()
    stack.push(node)
    paths.push([])

    while not stack.isEmpty():
            
        node = stack.pop()
        current_path = paths.pop()
        if(problem.isGoalState(node)):
            break

        if node not in visited:
            successors = problem.getSuccessors(node)
            visited.add(node)
            for neighbor in successors:
                if neighbor[0] not in visited:
                    stack.push(neighbor[0])
                    paths.push(current_path + [neighbor[1]])
            
    return current_path

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    paths = util.Queue()
    visited = set()
    node = problem.getStartState()
    queue.push(node)
    paths.push([])

    while not queue.isEmpty():
        node = queue.pop()
        current_path = paths.pop()

        if(problem.isGoalState(node)):
            break
        
        if node not in visited:
            successors = problem.getSuccessors(node)
            visited.add(node)
            for neighbor in successors:
                if neighbor[0] not in visited:
                    queue.push(neighbor[0])
                    paths.push(current_path + [neighbor[1]])
    
    return current_path

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    queue = util.PriorityQueue()
    paths = util.PriorityQueue()
    visited = set()
    queue.push(problem.getStartState(), 0)
    paths.push([], 0)

    while not queue.isEmpty():
        node = queue.pop()
        current_path = paths.pop()

        if problem.isGoalState(node):
            return current_path
        
        if node not in visited:
            successors = problem.getSuccessors(node)
            visited.add(node)
            for neighbor in successors:
                if neighbor[0] not in visited:
                    cost = problem.getCostOfActions(current_path + [neighbor[1]])
                    queue.push(neighbor[0], cost)
                    paths.push(current_path + [neighbor[1]], cost)
    
    return current_path  # Retorna uma lista vazia se não encontrar o objetivo


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
