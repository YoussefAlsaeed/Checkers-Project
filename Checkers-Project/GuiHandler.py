import pygame, sys
from pygame.locals import *
from time import sleep

pygame.font.init()

WHITE    = (255, 255, 255)
GREY     = (128, 128, 128)
PURPLE      = (178, 102, 255)
BLACK    = (  0,   0,   0)
GOLD     = (51,0,51)
HIGH     = (160, 190, 255)


"""
The Board class represents the game board and the pieces on it. It has methods for initializing the board, checking for legal moves, and moving pieces.
It also has methods for checking whether a piece should be kinged and for removing pieces from the board.
"""

class Board:
	def __init__(self):
		self.matrix = self.create_board()

	def create_board(self):
		# Create a list of 8 rows, each containing 8 squares with the appropriate color
		# If the sum of the row and column indices is even, set the square color to white
		# If the sum of the row and column indices is odd, set the square color to black
		matrix = [[Square(WHITE) if (x + y) % 2 == 0 else Square(BLACK) for x in range(8)] for y in range(8)]

		# Place the purple and grey pieces on the appropriate squares
		# For each row and column of the matrix, check if the square is black and in the first three or last three rows
		# If the square meets these criteria, place a purple or grey piece on it, respectively
		for x in range(8):
			for y in range(3):
				if matrix[x][y].color == BLACK:
					matrix[x][y].squarePiece = Piece(PURPLE)
			for y in range(5, 8):
				if matrix[x][y].color == BLACK:
					matrix[x][y].squarePiece = Piece(GREY)

		# Return the completed matrix
		return matrix


	def remove_piece(self, x, y):
		self.matrix[x][y].squarePiece = None

	def move_piece(self, start_x, start_y, end_x, end_y):

		self.matrix[end_x][end_y].squarePiece = self.matrix[start_x][start_y].squarePiece
		self.remove_piece(start_x, start_y)
		self.king(end_x, end_y)

	"""
	This method takes three arguments: the direction to move in (dir), and the current coordinates (x and y).
	The method then uses a series of if statements to determine the new coordinates based on the specified direction. 
	If the direction is valid, the method returns a tuple containing the new coordinates.
	If the direction is invalid, the method returns 0.
	"""
	def adjacent_square(self, dir, x, y):
		# If the direction is northwest, subtract 1 from both x and y to move to the new square
		if dir == "northwest":
			return (x - 1, y - 1)
		# If the direction is northeast, add 1 to x and subtract 1 from y to move to the new square
		elif dir == "northeast":
			return (x + 1, y - 1)
		# If the direction is southwest, subtract 1 from x and add 1 to y to move to the new square
		elif dir == "southwest":
			return (x - 1, y + 1)
		# If the direction is southeast, add 1 to both x and y to move to the new square
		elif dir == "southeast":
			return (x + 1, y + 1)
		# If the direction is invalid, return 0
		else:
			return 0

	# Return a list of the coordinates of the four squares adjacent to the one at (x,y)
	# The adjacent squares are obtained by calling the "rel" method with each of the four directions
	def getAdjacentSquares(self, x, y):
		return [self.adjacent_square("northwest", x, y), self.adjacent_square("northeast", x, y), self.adjacent_square("southwest", x, y), self.adjacent_square("southeast", x, y)]

	# Return the board square with those coordinates
	def getSquare(self, x, y):
		x = int(x)
		y = int(y)
		return self.matrix[x][y]

	"""
	If the piece is a grey non-king, the method returns a list of the two squares diagonally in front of it. 
	This is because grey pieces can only move diagonally forward. 
	If the piece is a purple non-king, the method returns a list of the two squares diagonally behind it. 
	This is because purple pieces can only move diagonally backward. 
	If the piece is a king, the method returns a list of all four adjacent squares, as kings can move in any direction diagonally.
	"""

	def get_legal_moves(self, x, y):
		if self.matrix[x][y].squarePiece != None:

			if self.matrix[x][y].squarePiece.king == False and self.matrix[x][y].squarePiece.color == GREY:
				blind_legal_moves = [self.adjacent_square("northwest", x, y), self.adjacent_square("northeast", x, y)]

			elif self.matrix[x][y].squarePiece.king == False and self.matrix[x][y].squarePiece.color == PURPLE:
				blind_legal_moves = [self.adjacent_square("southwest", x, y), self.adjacent_square("southeast", x, y)]

			else:
				blind_legal_moves = [self.adjacent_square("northwest", x, y), self.adjacent_square("northeast", x, y), self.adjacent_square("southwest", x, y), self.adjacent_square("southeast", x, y)]

		else:
			blind_legal_moves = []

		return blind_legal_moves

	def get_valid_legal_moves(self, x, y, continue_playing=False):
		# Get the blind legal moves for the specified square
		blind_legal_moves = self.get_legal_moves(x, y)

		# Initialize an empty list to store the valid legal moves
		legal_moves = []

		# If the player is not continuing a capture sequence
		if continue_playing == False:
			# Check each blind legal move
			for move in blind_legal_moves:
				# If the move is on the board
				if self.within_bounds(move[0], move[1]):
					# If the destination square is empty
					if self.getSquare(move[0], move[1]).squarePiece == None:
						# Add the move to the list of legal moves
						legal_moves.append(move)
					# If the destination square contains an enemy piece and the jump square is empty
					elif self.getSquare(move[0], move[1]).squarePiece.color != self.getSquare(x, y).squarePiece.color and self.within_bounds(move[0] + (move[0] - x), move[1] + (move[1] - y)) and self.getSquare(move[0] + (move[0] - x), move[1] + (move[1] - y)).squarePiece == None:
						# Add the jump move to the list of legalmoves
						legal_moves.append((move[0] + (move[0] - x), move[1] + (move[1] - y)))

		# If the player is continuing a capture sequence
		else:
			# Check each blind legal move
			for move in blind_legal_moves:
				# If the move is on the board and the destination square contains an enemy piece and the jump square is empty
				if self.within_bounds(move[0], move[1]) and self.getSquare(move[0], move[1]).squarePiece != None and self.getSquare(
						move[0], move[1]).squarePiece.color != self.getSquare(x, y).squarePiece.color and self.within_bounds(move[0] + (move[0] - x), move[1] + (move[1] - y)) and self.getSquare(move[0] + (move[0] - x), move[1] + (move[1] - y)).squarePiece == None:
					# Add the jump move to the list of legal moves
					legal_moves.append((move[0] + (move[0] - x), move[1] + (move[1] - y)))

		# Return the list of legal moves
		return legal_moves

	def within_bounds(self, x, y):

		if x < 0 or y < 0 or x > 7 or y > 7:
			return False
		else:
			return True


	def king(self, x, y):
		if self.getSquare(x, y).squarePiece != None:
			if (self.getSquare(x, y).squarePiece.color == GREY and y == 0) or (self.getSquare(x, y).squarePiece.color == PURPLE and y == 7):
				self.getSquare(x, y).squarePiece.crown()


class Piece:
	def __init__(self, color, king = False):
		self.color = color
		self.king = king
		self.value = 1

	def crown(self):
		self.king = True
		self.value = 2


