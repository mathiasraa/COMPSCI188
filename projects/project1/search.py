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


class Node:
    def __init__(self, val=None, action=None, parent=None, priority=0):
        self.val = val
        self.action = action
        self.parent = parent
        self.priority = priority

    def __str__(self) -> str:
        return str(self.val) + str(self.parent)

    def getPath(self):
        if not self.parent:
            return []

        return [self.action] + Node.getPath(self.parent)

    def getNodePriorities(self):
        if not self.parent:
            return []

        return [self.priority] + Node.getNodePriorities(self.parent)


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
        # util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        # util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        # util.raiseNotDefined()

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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    visited = {}
    stack = util.Stack()
    stack.push(Node(val=problem.getStartState()))

    while not stack.isEmpty():
        s = stack.pop()

        if problem.isGoalState(s.val):
            return Node.getPath(s)[::-1]

        if not visited.get(s.val):
            visited[s.val] = True

            for neighbor, action, _ in problem.getSuccessors(s.val):
                stack.push(Node(val=neighbor, action=action, parent=s))

    return None


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = {}
    stack = util.Queue()
    stack.push(Node(val=problem.getStartState()))

    while not stack.isEmpty():
        s = stack.pop()

        if problem.isGoalState(s.val):
            return Node.getPath(s)[::-1]

        if not visited.get(s.val):
            visited[s.val] = True

            for neighbor, action, _ in problem.getSuccessors(s.val):
                stack.push(Node(val=neighbor, action=action, parent=s))

    return []


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = {}
    stack = util.PriorityQueue()
    stack.push(Node(val=problem.getStartState()), priority=0)

    while not stack.isEmpty():
        s = stack.pop()

        if problem.isGoalState(s.val):
            return Node.getPath(s)[::-1]

        if not visited.get(s.val):
            visited[s.val] = True

            for neighbor, action, cost in problem.getSuccessors(s.val):
                stack.push(
                    Node(
                        val=neighbor,
                        action=action,
                        parent=s,
                        priority=s.priority + cost,
                    ),
                    priority=s.priority + cost,
                )

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visited = {}
    stack = util.PriorityQueue()
    start_state = problem.getStartState()
    stack.push(
        Node(val=start_state, priority=heuristic(start_state, problem)),
        heuristic(start_state, problem),
    )

    while not stack.isEmpty():
        s = stack.pop()

        if problem.isGoalState(s.val):
            return Node.getPath(s)[::-1]

        if not visited.get(s.val):
            visited[s.val] = True

            for neighbor, action, cost in problem.getSuccessors(s.val):
                stack.push(
                    Node(
                        val=neighbor,
                        action=action,
                        parent=s,
                        priority=cost + s.priority,
                    ),
                    priority=cost + s.priority + heuristic(neighbor, problem),
                )

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
