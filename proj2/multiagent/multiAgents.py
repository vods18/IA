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


from util import manhattanDistance
from game import Directions
import random, util
import math
from game import Agent
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
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        
        import math
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = currentGameState.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood()
        new_ghost_states = successor_game_state.getGhostStates()
        new_scared_times = [ghostState.scaredTimer for ghostState in new_ghost_states]
        scared_time = min(new_scared_times)

        current_food_list = currentGameState.getFood().asList()
        current_cap = currentGameState.getCapsules()

        new_food_list = new_food.asList()
        new_capsule_list = successor_game_state.getCapsules()

        # infinity positions to start
        closest_food_dist = math.inf 
        farthest_food_dist = -(math.inf)
        closest_cap_dist = math.inf
        closest_ghost = math.inf
        
        evaluation = 10000.0

        # found the closest food based on the next position
        for food in new_food_list:
            dist = manhattanDistance(new_pos, food)
            if dist < closest_food_dist and dist != 0:
                closest_food_dist = dist
                closest_food_position = food

        # if was found
        if closest_food_dist != math.inf:
            # here we will consider a score based on the distance of the closest food
            evaluation += 1000.0 / closest_food_dist  
            
            # found the farthest food based on the closest food
            for food in new_food_list:
                dist = manhattanDistance(food, closest_food_position)
                if dist > farthest_food_dist and dist != 0:
                    farthest_food_dist = dist

        # if was found
        if farthest_food_dist != math.inf:
            evaluation += 1000.0 / farthest_food_dist # here we will consider a score based on the distance of the farthest food

        # found the closest capsule based on the next position 
        for capsule in new_capsule_list:
            dist = manhattanDistance(capsule, new_pos)
            if dist < closest_cap_dist and dist != 0:
                closest_cap_dist = dist

        # if was found
        if closest_cap_dist != math.inf:
            evaluation += 1000.0 / closest_cap_dist

        for ghost in new_ghost_states:
            dist = manhattanDistance(ghost.getPosition(), new_pos)
            if dist < closest_ghost:
                closest_ghost = dist

        # if we have a cap closer then a ghost that is an advantage
        if closest_cap_dist != math.inf:
            if closest_cap_dist < closest_ghost:
                evaluation += closest_ghost
            else:
                evaluation -= closest_ghost
        
        if len(new_food_list) < len(current_food_list) or (len(new_capsule_list) < len(current_cap) and scared_time < 2):
            evaluation += 15000.0
        elif len(new_food_list) < len(current_food_list) or len(new_capsule_list) < len(current_cap):
            evaluation += 10000.0

        # if we have time to kill the ghost
        if scared_time > closest_ghost:
            evaluation += 10000.0

        if closest_ghost < 2 and scared_time < 2:
            evaluation -= 100000.0

        return evaluation
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        # Get legal moves for Pac-Man
        legalMoves = gameState.getLegalActions(0)
        
        # Generate future states for each legal move
        futureStates = [gameState.generateSuccessor(0, move) for move in legalMoves]

        # Calculate scores for each future state using minimizer
        scores = [self.minimizer(0, state, 1) for state in futureStates]
        
        # Find the best score and its indices
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        
        # Pick the first best position
        bestIndex = bestIndices[0]
        
        # Return the best action
        return legalMoves[bestIndex]

    def maximizer(self, currentDepth, gameState):
        # Check if the maximum depth is reached or the game is over
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        # Calculate the maximum score among minimizer's scores
        return max([self.minimizer(currentDepth, state, 1) for state in
                    [gameState.generateSuccessor(0, move) for move in gameState.getLegalActions(0)]])

    def minimizer(self, currentDepth, gameState, ghostIndex):
        # Check if the maximum depth is reached or the game is over
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        # If it's the last ghost, calculate the minimum score among maximizer's scores
        if ghostIndex + 1 >= gameState.getNumAgents():
            return min([self.maximizer(currentDepth + 1, state) for state in
                        [gameState.generateSuccessor(ghostIndex, move) for move in
                        gameState.getLegalActions(ghostIndex)]])

        # Calculate the minimum score among other minimizer's scores
        return min([self.minimizer(currentDepth, state, ghostIndex + 1) for state in
                    [gameState.generateSuccessor(ghostIndex, move) for move in
                    gameState.getLegalActions(ghostIndex)]])



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestScore = -math.inf  # Initialize the best value as negative infinity
        alpha = -math.inf  # Initialize alpha as negative infinity
        beta = math.inf  # Initialize beta as positive infinity
        actionSeq = []

        # Iterate through legal actions for Pac-Man (agent 0)
        for move in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, move)
            
            # Call minimax with alpha-beta pruning to get the value
            new_score = self.minimaxPrune(1, range(gameState.getNumAgents()), state, self.depth, self.evaluationFunction, alpha, beta)
            
            # Update the best value and action sequence
            if new_score > bestScore:
                bestScore = new_score
                actionSeq = move
            
            # Prune the search tree if val is greater than beta
            if bestScore > beta:
                return actionSeq
            
            # Update alpha
            alpha = max(alpha, bestScore)
        
        return actionSeq

    def minimaxPrune(self, agent, agents, state, depth, eval_function, alpha, beta):
        if depth <= 0 or state.isWin() or state.isLose():
            return eval_function(state)

        if agent == 0:
            bestScore = -math.inf  # Initialize value as negative infinity for max agent
        else:
            bestScore = math.inf  # Initialize value as positive infinity for min agents

        for move in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, move)
            
            if agent == agents[-1]:  # Last agent is a min agent
                bestScore = min(bestScore, self.minimaxPrune(agents[0], agents, successor, depth - 1, eval_function, alpha, beta))
                beta = min(beta, bestScore)
                if bestScore < alpha:
                    return bestScore
            elif agent == 0:  # Max agent
                bestScore = max(bestScore, self.minimaxPrune(agents[agent + 1], agents, successor, depth, eval_function, alpha, beta))
                alpha = max(alpha, bestScore)
                if bestScore > beta:
                    return bestScore
            else:  # Other min agents
                bestScore = min(bestScore, self.minimaxPrune(agents[agent + 1], agents, successor, depth, eval_function, alpha, beta))
                beta = min(beta, bestScore)
                if bestScore < alpha:
                    return bestScore
        
        return bestScore

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
