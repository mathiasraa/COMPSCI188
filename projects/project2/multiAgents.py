# multiAgents.py
# --------------
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


from math import inf
from statistics import mean
from typing import Tuple
from searchAgents import PositionSearchProblem
import search
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent, Actions
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food = currentGameState.getFood()

        distancesToFood = [
            abs(newPos[0] - foodPos[0]) + abs(newPos[1] - foodPos[1])
            for foodPos in food.asList()
        ]

        if action == Directions.STOP:
            return -inf

        if newPos in [s.getPosition() for s in newGhostStates]:
            return -inf

        return -min(distancesToFood or [inf])


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.minimax(gameState, self.depth, 0)["action"]

    def minimax(self, state: GameState, currDepth, agentIndex):
        if currDepth == 0 or state.isWin() or state.isLose():
            return {"val": self.evaluationFunction(state), "action": None}

        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = currDepth if nextAgent > 0 else currDepth - 1

        if agentIndex == 0:
            actionVal = {"val": -inf, "action": None}

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)

                v = self.minimax(successor, nextDepth, nextAgent)["val"]
                if v > actionVal["val"]:
                    actionVal = {"val": v, "action": action}

            return actionVal
        else:
            actionVal = {"val": inf, "action": None}

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)

                v = self.minimax(successor, nextDepth, nextAgent)["val"]

                if v < actionVal["val"]:
                    actionVal = {"val": v, "action": action}

            return actionVal


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        return self.minimax(gameState, self.depth, 0, -inf, inf)["action"]

    def minimax(self, state: GameState, currDepth, agentIndex, alpha, beta):
        if currDepth == 0 or state.isWin() or state.isLose():
            return {"val": self.evaluationFunction(state), "action": None}

        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = currDepth if nextAgent > 0 else currDepth - 1

        if agentIndex == 0:
            actionVal = {"val": -inf, "action": None}

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)

                v = self.minimax(successor, nextDepth, nextAgent, alpha, beta)["val"]
                if v > actionVal["val"]:
                    actionVal = {"val": v, "action": action}

                if v > beta:
                    return {"val": v, "action": action}

                if v > alpha:
                    alpha = v

            return actionVal
        else:
            actionVal = {"val": inf, "action": None}

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)

                v = self.minimax(successor, nextDepth, nextAgent, alpha, beta)["val"]

                if v < actionVal["val"]:
                    actionVal = {"val": v, "action": action}

                if v < alpha:
                    return {"val": v, "action": action}

                if v < beta:
                    beta = v

            return actionVal


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        scores = [
            self.minimax(
                gameState.generateSuccessor(0, action),
                self.depth,
                1,
            )
            for action in legalMoves
        ]
        maxScoreIndex = max(range(len(scores)), key=scores.__getitem__)
        return legalMoves[maxScoreIndex]

    def minimax(self, state: GameState, currDepth, agentIndex):
        if currDepth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        minmaxFunc = max if agentIndex == 0 else mean
        nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
        nextDepth = (currDepth - 1) if nextAgentIndex == 0 else currDepth
        legalMoves = state.getLegalActions(agentIndex)
        nextStates = [
            state.generateSuccessor(agentIndex, action) for action in legalMoves
        ]
        scores = [
            self.minimax(nextState, nextDepth, nextAgentIndex)
            for nextState in nextStates
        ]

        return minmaxFunc(scores)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Food value is calculated by inverting the value of all distances to food
                 (smaller distance means increased value),
                 and then summing this. Ghost value is the same concept, only magnified by 500.
                 This makes pacman prioritize to eat ghosts that have a zero scaredTimer.
    """
    "*** YOUR CODE HERE ***"
    currentScore = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    distGhosts = [
        manhattanDistance(pos, g.getPosition()) for g in ghosts if g.scaredTimer != 0
    ]

    distFood = sorted([(manhattanDistance(pos, food), food) for food in food.asList()])
    distFood = [mazeDistance(pos, food, currentGameState) for d, food in distFood[:6]]
    foodValue = sum(1 / d for d in distFood) if distFood else 0

    distCapsules = [manhattanDistance(pos, g) for g in capsules]
    capsulesValue = sum((1 / cap) for cap in distCapsules)
    ghostValue = sum(((1 / dist) * 500 for dist in distGhosts if dist != 0)) * 0.8

    return currentScore * 1.4 + foodValue + ghostValue + capsulesValue


def mazeDistance(
    point1: Tuple[int, int], point2: Tuple[int, int], gameState: GameState
) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], "point1 is a wall: " + str(point1)
    assert not walls[x2][y2], "point2 is a wall: " + str(point2)
    prob = PositionSearchProblem(
        gameState, start=point1, goal=point2, warn=False, visualize=False
    )
    return len(search.bfs(prob))


# Abbreviation
better = betterEvaluationFunction
