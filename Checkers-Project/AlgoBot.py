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

