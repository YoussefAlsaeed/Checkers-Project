import pygame
from pygame.locals import *
import random
from copy import deepcopy
import math
pygame.font.init()

GREY = (128, 128, 128)
PURPLE = (178, 102, 255)
BLACK = (0,   0,   0)

class Bot:
    def __init__(self, game, color, method='random', depth=1):
        # Initialize the attributes of the Player object
        self.method = method
        self.mid_phase = self.evaluate  # Set the evaluation function for the mid-game phase
        self.end_phase = self.evaluateDistance  # Set the evaluation function for the end-game phase
        self.depth = depth  # Set the depth of the algorithm
        self.game = game  # Set the Game object that the Player is playing
        self.color = color  # Set the color of the Player's pieces
        self.eval_color = color  # Set the color used for evaluation (initially set to the Player's color)
        if self.color == GREY:
            self.opponent_color = PURPLE  # Set the color of the opponent's pieces
        else:
            self.opponent_color = GREY  # Set the color of the opponent's pieces
        self._current_eval = self.mid_phase  # Initialize the current evaluation function to the mid-game evaluation
        self._end_eval_time = False  # Initialize the end-game evaluation time to False

    def iskings(self, board):
        # Check if all pieces on the board are kings
        for i in range(8):
            for j in range(8):
                squarePiece = board.getSquare(i, j).squarePiece
                if squarePiece is not None and squarePiece.king == False:
                    return False
        return True

    def distance(self, x1, y1, x2, y2):
        # Calculate the Euclidean distance between two points
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    """
    The step method checks if the game has entered the end phase, and if so, evaluates the current board state.
    It then calls the appropriate move method based on the method attribute of the class instance.
    """
    def step(self, board):
        if self.end_phase is not None and self._end_eval_time == False:
            if self.iskings(board):
                self._end_eval_time = True
                self._current_eval = self.end_phase
        if self.method == 'random':
            self.random_step(board)
        elif self.method == 'minmax':
            self.minmax_step(board)
        elif self.method == 'alpha_beta':
            self.alpha_beta_step(board)

    def getPossibleMoves(self, board):
        possible_moves = []
        for i in range(8):
            for j in range(8):
                if(board.get_valid_legal_moves(i, j, self.game.continue_playing) != [] and board.getSquare(i, j).squarePiece != None and board.getSquare(i, j).squarePiece.color == self.game.turn):
                    possible_moves.append((i, j, board.get_valid_legal_moves(i, j, self.game.continue_playing)))
        return possible_moves

    def generatemove_at_a_time(self, board):
        for i in range(8):
            for j in range(8):
                if(board.get_valid_legal_moves(i, j, self.game.continue_playing) != [] and board.getSquare(i, j).squarePiece != None and board.getSquare(i, j).squarePiece.color == self.game.turn):
                    yield (i, j, board.get_valid_legal_moves(i, j, self.game.continue_playing))

    def random_step(self, board):
        possible_moves = self.getPossibleMoves(board) # get all possible moves for current piece
        if possible_moves == []:     # end turn if there are no possible moves
            self.game.end_turn()
            return
        random_move = random.choice(possible_moves)
        rand_choice = random.choice(random_move[2])   # choose a random move from the choices
        self.move(random_move, rand_choice, board)  # call action method to move
        return

    def minmax_step(self, board):
        random_move, random_choice, _ = self.minimax(
            self.depth - 1, board, 'max')
        self.move(random_move, random_choice, board)
        return

    def alpha_beta_step(self, board):
        random_move, random_choice, _ = self.alphaBeta(self.depth - 1, board, 'max', alpha=-float('inf'), beta=float('inf'))
        self.move(random_move, random_choice, board)
        return

    def move(self, current_pos, final_pos, board):
        # If the current position is None, end the turn
        if current_pos is None:
            self.game.end_turn()

        # If the game is over, check if the final position is occupied by a friendly piece
        if self.game.continue_playing == False:
            if board.getSquare(final_pos[0], final_pos[1]).squarePiece is not None and board.getSquare(final_pos[0],
                                                                                                       final_pos[
                                                                                                           1]).squarePiece.color == self.game.turn:
                # If it is, update the current position
                current_pos = final_pos

            # Otherwise, if the final position is a valid legal move from the current position, make the move
            elif current_pos != None and final_pos in board.get_valid_legal_moves(current_pos[0], current_pos[1]):
                board.move_piece(
                    current_pos[0], current_pos[1], final_pos[0], final_pos[1])

                # If the move is a capture, remove the captured piece and allow the player to continue making moves
                if final_pos not in board.getAdjacentSquares(current_pos[0], current_pos[1]):
                    board.remove_piece(current_pos[0] + (final_pos[0] - current_pos[0]) //
                                       2, current_pos[1] + (final_pos[1] - current_pos[1]) // 2)
                    self.game.continue_playing = True

                # Update the current position and check if the player can make another move
                current_pos = final_pos
                final_pos = board.get_valid_legal_moves(
                    current_pos[0], current_pos[1], True)
                if final_pos != []:
                    self.move(current_pos, final_pos[0], board)

                # End the turn
                self.game.end_turn()

        # If the player can continue making moves, check if the final position is a valid legal move
        if self.game.continue_playing == True:
            if current_pos != None and final_pos in board.get_valid_legal_moves(current_pos[0], current_pos[1],
                                                                                self.game.continue_playing):
                board.move_piece(
                    current_pos[0], current_pos[1], final_pos[0], final_pos[1])

                # If the move is a capture, remove the captured piece
                board.remove_piece(current_pos[0] + (final_pos[0] - current_pos[0]) //
                                   2, current_pos[1] + (final_pos[1] - current_pos[1]) // 2)

            # Check if the player can make another move or if the turn should end
            if board.get_valid_legal_moves(final_pos[0], final_pos[1], self.game.continue_playing) == []:
                self.game.end_turn()
            else:
                # Update the current position and check if the player can make another move
                current_pos = final_pos
                final_pos = board.get_valid_legal_moves(
                    current_pos[0], current_pos[1], True)
                if final_pos != []:
                    self.move(current_pos, final_pos[0], board)

                # End the turn
                self.game.end_turn()

        # If the game is over, switch to the opponent's turn
        if self.game.continue_playing != True:
            self.game.turn = self.opponent_color

    def moveOnBoard(self, board, current_pos, final_pos, continue_playing=False):
        # If continue_playing is False, check if the final position is occupied by a friendly piece
        if continue_playing == False:
            if board.getSquare(final_pos[0], final_pos[1]).squarePiece != None and board.getSquare(final_pos[0],
                                                                                                   final_pos[1]).squarePiece.color == self.game.turn:
                # If it is, update the current position
                current_pos = final_pos

            # Otherwise, if the final position is a valid legal move from the current position, make the move
            elif current_pos != None and final_pos in board.get_valid_legal_moves(current_pos[0], current_pos[1]):
                board.move_piece(
                    current_pos[0], current_pos[1], final_pos[0], final_pos[1])

                # If the move is a capture, remove the captured piece and allow the player to continue making moves
                if final_pos not in board.getAdjacentSquares(current_pos[0], current_pos[1]):
                    board.remove_piece(current_pos[0] + (final_pos[0] - current_pos[0]) //
                                       2, current_pos[1] + (final_pos[1] - current_pos[1]) // 2)
                    continue_playing = True

                # Update the current position and# If continue_playing is False, check if the final position is occupied by a friendly piece
                current_pos = final_pos
                final_pos = board.get_valid_legal_moves(current_pos[0], current_pos[1], True)
                if final_pos != []:
                    # Recursively call moveOnBoard with the new final_pos and continue_playing set to True
                    self.moveOnBoard(board, current_pos, final_pos[0], continue_playing=True)

        # If continue_playing is True, check if the final position is a valid legal move
        else:
            if current_pos != None and final_pos in board.get_valid_legal_moves(current_pos[0], current_pos[1],continue_playing):
                board.move_piece(current_pos[0], current_pos[1], final_pos[0], final_pos[1])

                # If the move is a capture, remove the captured piece
                board.remove_piece(current_pos[0] + (final_pos[0] - current_pos[0]) // 2,
                                   current_pos[1] + (final_pos[1] - current_pos[1]) // 2)

            # Check if the player can make another move or if the turn should end
            if board.get_valid_legal_moves(final_pos[0], final_pos[1], self.game.continue_playing) == []:
                return
            else:
                # Update the current position and recursively call moveOnBoard with the new final_pos and continue_playing set to True
                current_pos = final_pos
                final_pos = board.get_valid_legal_moves(current_pos[0], current_pos[1], True)
                if final_pos != []:
                    self.moveOnBoard(board, current_pos, final_pos[0], continue_playing=True)

    def minimax(self, depth, board, fn):
        # The minimax function is a recursive function that searches through
        # the game tree to find the best move for the current player.
        # It takes the current game state, the maximum depth to search,
        # and the current player as inputs.

        # If we have reached the maximum depth or if the game is over,
        # we evaluate the current position and return the result.
        if depth == 0:
            if fn == 'max':
                # If we are at a max node, we want to find the best move
                # that maximizes our score.
                max_value = -float("inf")
                bestLocation = None
                bestMove = None
                for pos in self.generatemove_at_a_time(board):
                    # Generate all possible moves for the current player
                    for move in pos[2]:
                        # Try each move
                        board_clone = deepcopy(board)
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        self.moveOnBoard(board_clone, pos, move)
                        # Evaluate the resulting board position
                        currentValue = self._current_eval(board_clone)
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color

                        # Update the best move if this move is better
                        if currentValue > max_value:
                            max_value = currentValue
                            bestLocation = pos
                            bestMove = (move[0], move[1])
                        # If multiple moves have the same score, randomly choose one
                        elif currentValue == max_value and random.random() <= 0.5:
                            max_value = currentValue
                            bestLocation = (pos[0], pos[1])
                            bestMove = (move[0], move[1])
                        # If we have found a winning move, choose it immediately
                        if(currentValue == -float("inf") and bestLocation is  None):
                            bestLocation = (pos[0], pos[1])
                            bestMove = (move[0], move[1])
                # Return the best move and its score
                return bestLocation, bestMove, max_value
            else:
                # If we are at a min node, we want to find the best move
                # that minimizes our opponent's score.
                min_value = float("inf")
                bestLocation = None
                bestMove = None
                for pos in self.generatemove_at_a_time(board):
                    # Generate all possible moves for the current player
                    for move in pos[2]:
                        board_clone = deepcopy(board)
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        self.moveOnBoard(board_clone, pos, move)
                        # Evaluate the resulting board position
                        currentValue = self._current_eval(board_clone)
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        # Update the best move if this move is better for our opponent
                        if currentValue < min_value:
                            min_value = currentValue
                            bestLocation = pos
                            bestMove = move
                        # If multiple moves have the same score, randomly choose one
                        elif currentValue == min_value and random.random() <= 0.5:
                            min_value = currentValue
                            bestLocation = pos
                            bestMove = move
                        # If we have found a losing move, choose it immediately
                        if(currentValue == float("inf") and bestLocation is  None):
                            bestLocation = (pos[0], pos[1])
                            bestMove = (move[0], move[1])
                # Return the best move and its score
                return bestLocation, bestMove, min_value
        else:
            if fn == 'max':
                # If we are at a max node, we want to find the best move
                # that maximizes our score, considering all possible opponent responses.
                max_value = -float("inf")
                bestLocation = None
                bestMove = None
                for pos in self.generatemove_at_a_time(board):
                    # Generate all possible moves for the current player
                    for move in pos[2]:
                        # Try each move
                        board_clone = deepcopy(board)
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        self.moveOnBoard(board_clone, pos, move)
                        # If the game is over, return the result immediately
                        if self.endGameCheck(board_clone):
                            currentValue = float("inf")
                        else:
                            # Otherwise, call minimax recursively from the opponent's perspective
                            _, _, currentValue = self.minimax(depth - 1, board_clone, 'min')
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        if(currentValue is None):
                            continue
                        # Update the best move if this move is better
                        if currentValue > max_value:
                            max_value = currentValue
                            bestLocation = pos
                            bestMove = move
                        # If multiple moves have the same score, randomly choose one
                        elif currentValue == max_value and random.random() <= 0.5:
                            max_value = currentValue
                            bestLocation = pos
                            bestMove = move
                        # If we have found a winning move, choose it immediately
                        if(currentValue == -float("inf") and bestLocation is  None):
                            bestLocation = (pos[0], pos[1])
                            bestMove = (move[0], move[1])
                # Return the best move and its score
                return bestLocation, bestMove, max_value
            else:
                # If we are at a min node, we want to find the best move
                # that minimizes our opponent's score, considering all possible opponent responses.
                min_value = float("inf")
                bestLocation = None
                bestMove = None
                for pos in self.generatemove_at_a_time(board):
                    for move in pos[2]:
                        board_clone = deepcopy(board)
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        self.moveOnBoard(board_clone, pos, move)
                        # If the game is over, return the result immediately
                        if self.endGameCheck(board_clone):
                            currentValue = -float("inf")
                        else:
                            # Otherwise, call minimax recursively from our perspective
                            _, _, currentValue = self.minimax(depth - 1, board_clone, 'max')
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        if(currentValue is None):
                            continue
                        # Update the best move if this move is better for our opponent
                        if currentValue < min_value:
                            min_value = currentValue
                            bestLocation = (pos[0], pos[1])
                            bestMove = (move[0], move[1])
                        # If multiple moves have the same score, randomly choose one
                        elif currentValue == min_value and random.random() <= 0.5:
                            min_value = currentValue
                            bestLocation = pos
                            bestMove = move
                        # If we have found a losing move, choose it immediately
                        if(currentValue == float("inf") and bestLocation is  None):
                            bestLocation = (pos[0], pos[1])
                            bestMove = (move[0], move[1])
                # Return the best move and its score
                return bestLocation, bestMove, min_value

    """
        Implementation of the alpha-beta pruning algorithm for a two-player checkers game.

        Parameters:
            depth (int): The maximum depth to search the game tree.
            board (list): The current state of the game board.
            fn (str): The name of the function to maximize ('max') or minimize ('min') the value of.
            alpha (float): The best value that the maximizing player can guarantee at this point or better.
            beta (float): The best value that the minimizing player can guarantee at this point or worse.

        Returns:
            tuple: A tuple containing the best position, action, and value.
        """
    def alphaBeta(self, depth, board, fn, alpha, beta):
        # Base case: If depth is zero, return the best position, action, and value using the current evaluation function
        if depth == 0:
            if fn == 'max':
                max_value = -float("inf")
                best_pos = None
                best_action = None
                # Generate all possible moves for the current player and loop through them
                for pos in self.generatemove_at_a_time(board):
                    for action in pos[2]:
                        # Make a copy of the board and simulate the move
                        board_clone = deepcopy(board)
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        self.moveOnBoard(board_clone, pos, action)
                        # Evaluate the resulting board state
                        moveValue = self._current_eval(board_clone)
                        # Undo the move and switch back the player turn
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        # If the evaluated value is better than the current max value, update the best position, action, and value
                        if moveValue > max_value:
                            max_value = moveValue
                            best_pos = pos
                            best_action = (action[0], action[1])
                        # If the evaluated value is the same as the current max value, randomly choose between the two options
                        elif moveValue == max_value and random.random() <= 0.5:
                            max_value = moveValue
                            best_pos = (pos[0], pos[1])
                            best_action = (action[0], action[1])
                        # If the evaluated value is -inf and the best position is still None, update the best position and action
                        if(moveValue == -float("inf") and best_pos is  None):
                            best_pos = (pos[0], pos[1])
                            best_action = (action[0], action[1])
                        # Update the alpha value
                        alpha = max(alpha, max_value)
                        # If beta is less than or equal to alpha, prune the remaining branches.
                        if beta < alpha:
                            break
                    if beta < alpha:
                        break
                    # Return the best position, action, and value
                return best_pos, best_action, max_value
            # If the current function is 'min', do the same as above, but with the minimum value instead of the maximum value
            else:
                min_value = float("inf")
                best_pos = None
                best_action = None
                for pos in self.generatemove_at_a_time(board):
                    for action in pos[2]:
                        board_clone = deepcopy(board)
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        self.moveOnBoard(board_clone, pos, action)
                        moveValue = self._current_eval(board_clone)
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        if moveValue < min_value:
                            min_value = moveValue
                            best_pos = pos
                            best_action = action
                        elif moveValue == min_value and random.random() <= 0.5:
                            min_value = moveValue
                            best_pos = pos
                            best_action = action
                        if(moveValue == float("inf") and best_pos is  None):
                            best_pos = (pos[0], pos[1])
                            best_action = (action[0], action[1])
                        beta = min(beta, min_value)
                        if beta < alpha:

                            break
                    if beta < alpha:
                        break
                return best_pos, best_action, min_value
        else:
            if fn == 'max':
                max_value = -float("inf")
                best_pos = None
                best_action = None
                for pos in self.generatemove_at_a_time(board):
                    for action in pos[2]:
                        board_clone = deepcopy(board)
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        self.moveOnBoard(board_clone, pos, action)
                        # If the game has ended, set the step value to infinity.
                        if self.endGameCheck(board_clone):
                            moveValue = float("inf")
                        else:
                            # Recursively call the alphaBeta function with a decreased depth and the AI player's turn.
                            _, _, moveValue = self.alphaBeta(depth - 1, board_clone, 'min', alpha, beta)
                        # Switch back to the AI player's turn and update the best value and move if necessary.
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color

                        if(moveValue is None):
                            continue
                        if moveValue > max_value:
                            max_value = moveValue
                            best_pos = pos
                            best_action = action
                        # If the current value is equal to the maximum value, randomly choose between the two moves.
                        elif moveValue == max_value and random.random() <= 0.5:
                            max_value = moveValue
                            best_pos = pos
                            best_action = action
                        # If the step value is negative infinity, set the best move to the current move.
                        if(moveValue == -float("inf") and best_pos is  None):
                            best_pos = (pos[0], pos[1])
                            best_action = (action[0], action[1])
                        alpha = max(alpha, max_value)
                        # If beta is less than or equal to alpha, prune the remaining branches.
                        if beta <= alpha:
                            break
                    # If beta is less than or equal to alpha, prune the remaining branches.
                    if beta < alpha:
                        break
                return best_pos, best_action, max_value
            else:
                min_value = float("inf")
                best_pos = None
                best_action = None
                for pos in self.generatemove_at_a_time(board):
                    for action in pos[2]:
                        board_clone = deepcopy(board)
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        self.moveOnBoard(board_clone, pos, action)
                        if self.endGameCheck(board_clone):
                            moveValue = -float("inf")
                        else:
                            # Recursively call the alphaBeta function with a decreased depth and the AI player's turn.
                            _, _, moveValue = self.alphaBeta(depth - 1, board_clone, 'max', alpha, beta)
                        # Switch back to the opponent's turn and update the best value and move if necessary.
                        self.color, self.opponent_color = self.opponent_color, self.color
                        self.game.turn = self.color
                        if(moveValue is None):
                            continue
                        if moveValue < min_value:
                            min_value = moveValue
                            best_pos = (pos[0], pos[1])
                            best_action = (action[0], action[1])
                        # If the current value is equal to the minimum value, randomly choose between the two moves.
                        elif moveValue == min_value and random.random() <= 0.5:
                            min_value = moveValue
                            best_pos = pos
                            best_action = action
                        # If the step value is positive infinity, set the best move to the current move.
                        if(moveValue == float("inf") and best_pos is  None):
                            best_pos = (pos[0], pos[1])
                            best_action = (action[0], action[1])
                        beta = min(beta, min_value)
                        # If beta is less than alpha, prune the remaining branches.
                        if beta < alpha:
                            break
                    # If beta is less than alpha, prune the remaining branches.
                    if beta < alpha:
                        break
                return best_pos, best_action, min_value

    def evaluate(self, board):
        score = 0
        num_pieces = 0
        # Evaluate the board position based on the current player's color
        if (self.eval_color == PURPLE):
            for i in range(8):
                for j in range(8):
                    squarePiece = board.getSquare(i, j).squarePiece
                    if (squarePiece is not None):
                        num_pieces += 1
                        # Evaluate the score based on the type and position of the piece
                        if squarePiece.color == self.eval_color and squarePiece.king:
                            score += 10
                        elif squarePiece.color != self.eval_color and squarePiece.king:
                            score -= 10
                        elif squarePiece.color == self.eval_color and j < 4:
                            score += 5
                        elif squarePiece.color != self.eval_color and j < 4:
                            score -= 7
                        elif squarePiece.color == self.eval_color and j >= 4:
                            score += 7
                        elif squarePiece.color != self.eval_color and j >= 4:
                            score -= 5
        else:
            for i in range(8):
                for j in range(8):
                    squarePiece = board.getSquare(i, j).squarePiece
                    if (squarePiece is not None):
                        num_pieces += 1
                        # Evaluate the score based on the type and position of the piece
                        if squarePiece.color == self.eval_color and squarePiece.king:
                            score += 10
                        elif squarePiece.color != self.eval_color and squarePiece.king:
                            score -= 10
                        elif squarePiece.color == self.eval_color and j < 4:
                            score += 7
                        elif squarePiece.color != self.eval_color and j < 4:
                            score -= 5
                        elif squarePiece.color == self.eval_color and j >= 4:
                            score += 7
                        elif squarePiece.color != self.eval_color and j >= 4:
                            score -= 5
        # Return the average score per piece on the board
        return score / num_pieces

    def allPiecesLocation(self, board):
        """
        Returns the locations of all pieces on the board for the current player and the opponent.

        Parameters:
        board (Board): The current state of the game board.

        Returns:
        tuple: A tuple of two lists, one containing the locations of the current player's pieces and one containing
               the locations of the opponent's pieces.
        """
        # Initialize empty lists to store the locations of the current player's pieces and the opponent's pieces.
        player_pieces = []
        opponent_pieces = []

        # Iterate over each square on the board.
        for i in range(8):
            for j in range(8):
                # Get the piece on the current square, if any.
                squarePiece = board.getSquare(i, j).squarePiece
                if (squarePiece is not None):
                    # If there is a piece on the current square, add its location to the appropriate list
                    # based on its color.
                    if (squarePiece.color == self.eval_color):
                        player_pieces.append((i, j))
                    else:
                        opponent_pieces.append((i, j))

        # Return a tuple containing the lists of the current player's pieces and the opponent's pieces.
        return player_pieces, opponent_pieces

    def evaluateDistance(self, board):
        # Calculate the sum of distances between all player pieces and opponent pieces on the board
        player_pieces, adversary_pieces = self.allPiecesLocation(board)
        sum_of_dist = 0
        for pos in player_pieces:
            for adv in adversary_pieces:
                sum_of_dist += self.distance(pos[0], pos[1], adv[0], adv[1])
        # If the player has more pieces than the opponent, negate the sum of distances
        if (len(player_pieces) >= len(adversary_pieces)):
            sum_of_dist *= -1
        return sum_of_dist

    # Determine whether the current board position corresponds to an endgame phase
    def endGameCheck(self, board):
        # Iterate over every square on the board
        for x in range(8):
            for y in range(8):
                # Check whether the square is occupied by a piece of the player whose turn it is
                if board.getSquare(x, y).color == BLACK and board.getSquare(x,y).squarePiece is not None and board.getSquare(x, y).squarePiece.color == self.game.turn:
                    # Check whether the piece has any legal moves available
                    if board.get_valid_legal_moves(x, y) != []:
                        # If the piece has legal moves available, the game is not in an endgame phase
                        return False
        # If no player piece has legal moves available, the game is in an endgame phase
        return True