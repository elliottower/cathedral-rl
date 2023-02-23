from .piece import Piece


def get_pieces(color):
    if color == 0:
        return [
            Tavern1(),
            Tavern2(),
            Stable1(),
            Stable2(),
            Inn1(),
            Inn2(),
            Bridge(),
            Square(),
            Manor(),
            Abbey(),
            Academy(),
            Infirmary(),
            Castle(),
            Tower(),
            Cathedral(),
        ]
    elif color == 1:
        return [
            Tavern1(),
            Tavern2(),
            Stable1(),
            Stable2(),
            Inn1(),
            Inn2(),
            Bridge(),
            Square(),
            Manor(),
            AbbeyFlipped(),
            AcademyFlipped(),
            Infirmary(),
            Castle(),
            Tower(),
        ]


# Single piece
# [x]
class Tavern1(Piece):
    label = "Tavern1"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y)]

    def set_rotation(self, degree):
        if degree in [0, 90, 180, 270]:
            self.rotation = degree
        return


# Single piece
# [x]
class Tavern2(Piece):
    label = "Tavern2"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y)]
        self.rotation = 0

    def set_rotation(self, degree):
        if degree in [0, 90, 180, 270]:
            self.rotation = degree
        return


# Double piece
# [ ]
# [x]
class Stable1(Piece):
    label = "Stable1"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x, y + 1)]
        self.rotation = 0

    def set_rotation(self, degree):
        if degree in [0, 90, 180, 270]:
            # Reset piece
            x, y = self.position
            self.set_position(x, y)

            if degree in [90, 270]:
                self.rotate()
            return


# Double piece
# [ ]
# [x]
class Stable2(Piece):
    label = "Stable2"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x, y + 1)]
        self.rotation = 0

    def set_rotation(self, degree):
        if degree in [0, 90, 180, 270]:
            # Reset piece
            x, y = self.position
            self.set_position(x, y)

            if degree in [90, 270]:
                self.rotate()
            return


# Triple piece
# [ ]
# [x]
# [ ]
class Bridge(Piece):
    label = "Bridge"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y - 1)]
        self.rotation = 0

    def set_rotation(self, degree):
        if degree in [0, 90, 180, 270]:
            # Reset piece
            x, y = self.position
            self.set_position(x, y)

            if degree in [90, 270]:
                self.rotate()
            return


# L shape
# [ ]
# [x][ ]
class Inn1(Piece):
    label = "Inn1"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y)]
        self.rotation = 0


# L shape
# [ ]
# [x][ ]
class Inn2(Piece):
    label = "Inn2"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y)]
        self.rotation = 0


# Square shape
# [ ][ ]
# [x][ ]
class Square(Piece):
    label = "Square"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x + 1, y + 1)]
        self.rotation = 0

    def set_rotation(self, degree):
        if degree in [0, 90, 180, 270]:
            self.rotation = degree
        return


# T shape
#    [ ]
# [ ][x][ ]
class Manor(Piece):
    label = "Manor"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x - 1, y), (x + 1, y), (x, y + 1)]
        self.rotation = 0


# Z shape
#    [ ][ ]
# [ ][x]
class Abbey(Piece):
    label = "Abbey"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x - 1, y), (x, y + 1), (x + 1, y + 1)]
        self.rotation = 0


# Z shape (flipped)
# [ ][ ]
#    [x][ ]
class AbbeyFlipped(Piece):
    label = "AbbeyFlipped"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x - 1, y), (x, y + 1), (x + 1, y + 1)]
        self.rotation = 0


#       [ ]
# [ ][x][ ]
#    [ ]
class Academy(Piece):
    label = "Academy"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x + 1, y + 1)]
        self.rotation = 0


# [ ]
# [ ][x][ ]
#    [ ]
class AcademyFlipped(Piece):
    label = "AcademyFlipped"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x + 1, y + 1)]
        self.rotation = 0


# Plus shape
#    [ ]
# [ ][x][ ]
#    [ ]
class Infirmary(Piece):
    label = "Infirmary"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        self.rotation = 0

    def set_rotation(self, degree):
        if degree in [0, 90, 180, 270]:
            self.rotation = degree
        return


# U Shape
# [ ]   [ ]
# [ ][x][ ]
class Castle(Piece):
    label = "Castle"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x - 1, y), (x + 1, y), (x - 1, y + 1), (x + 1, y + 1)]
        self.rotation = 0


# W Shape
# [ ][ ]
#    [x][ ]
#       [ ]
class Tower(Piece):
    label = "Tower"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [(x, y), (x + 1, y), (x + 1, y - 1), (x, y + 1), (x - 1, y + 1)]
        self.rotation = 0


# Tall cross shape
#    [ ]
# [ ][ ][ ]
#    [x]
#    [ ]
class Cathedral(Piece):
    label = "Cathedral"

    def __init__(self):
        super().__init__()
        self.set_unplaced()

    def set_position(self, x, y):
        self.position = (x, y)
        self.points = [
            (x, y),
            (x, y - 1),
            (x, y + 1),
            (x - 1, y + 1),
            (x + 1, y + 1),
            (x, y + 2),
        ]
        self.rotation = 0
