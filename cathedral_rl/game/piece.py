import numpy as np


class Piece:
    """
    Base class for Cathedral pieces, with functions to rotate and flip pieces
    """

    def __init__(self):
        self.rotation_matrix = np.array([(0, -1), (1, 0)])
        self.rotation_matrix_ccw = np.array([(0, 1), (-1, 0)])
        self.position = (0, 0)
        self.points = []
        self.rotation = 0
        self.placed = False
        self.removed = False
        self.name = self.__class__.__name__

    @property
    def size(self):
        return len(self.points)

    def set_position(self, x, y):
        raise NotImplementedError()

    def set_rotation(self, degree):
        if degree in [0, 90, 180, 270]:
            # Reset piece
            x, y = self.position
            self.set_position(x, y)

            # Rotate piece
            rotations = int(degree / 90)
            for i in range(rotations):
                self.rotate()

    def set_placed(self):
        self.placed = True

    def is_placed(self):
        return self.placed

    def set_unplaced(self):
        self.set_position(-100, -100)
        self.placed = False

    def is_unplaced(self):
        return self.position == (-100, -100) and not self.placed and not self.removed

    def remove(self):
        self.set_position(-100, -100)
        self.placed = False
        self.removed = True

    def is_removed(self):
        return self.removed

    def rotate(self):
        # """
        # Returns the points that would be covered by a
        # shape that is rotated 0, 90, 180, of 270 degrees
        # in a clockwise direction.
        # """
        self.rotation = (self.rotation + 90) % 360

        reference = np.array([self.position])
        points = np.array(self.points)

        points = (points - reference) @ self.rotation_matrix + reference
        self.points = list(map(tuple, points))

    def rotate_ccw(self):
        # """
        # Returns the points that would be covered by a
        # shape that is rotated 0, 90, 180, of 270 degrees
        # in a counter-clockwise direction.
        # """
        self.rotation = (self.rotation - 90) % 360

        reference = np.array([self.position])
        points = np.array(self.points)

        points = (points - reference) @ self.rotation_matrix_ccw + reference
        self.points = list(map(tuple, points))
