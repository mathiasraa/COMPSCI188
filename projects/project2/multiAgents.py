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

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentScore = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()

    ghostDistances = [
        manhattanDistance(g, pos) for g in currentGameState.getGhostPositions()
    ]
    minGhostIndex = min(range(len(ghostDistances)), key=ghostDistances.__getitem__)

    closestGhost = currentGameState.getGhostStates()[minGhostIndex]

    # if currentGameState.isLose() and closestGhost.scaredTimer == 0:
    #     return -inf
    # if currentGameState.isWin():
    #     return 10000

    food = currentGameState.getFood()

    ghostBonus = (
        (1 / ghostDistances[minGhostIndex]) ** 100
        if closestGhost.scaredTimer != 0
        else 0
    )

    ghosts = currentGameState.getGhostStates()
    ghostDistances = [
        manhattanDistance(pos, tuple(map(int, ghost.configuration.pos)))
        for ghost in ghosts
    ]
    scaredTimers = [ghost.scaredTimer for ghost in ghosts]

    distFromScared = [
        dist for dist, timer in zip(ghostDistances, scaredTimers) if timer > 2
    ]
    ghostBonus = sum((190 / dist for dist in distFromScared), 0)

    foods = food.asList()
    manhattanDistances = [(manhattanDistance(pos, food), food) for food in foods]
    manhattanNearestFood = [food for dist, food in sorted(manhattanDistances)[:6]]
    mazeNearestFood = sorted(
        manhattanDistance(pos, food) for food in manhattanNearestFood
    )
    foodBonus = sum(1 / d for d in mazeNearestFood) if mazeNearestFood else 0

    return currentScore + foodBonus + ghostBonus


# Abbreviation
better = betterEvaluationFunction
