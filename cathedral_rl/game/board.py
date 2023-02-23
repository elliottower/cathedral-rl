import numpy as np

from .pieces import get_pieces


class Board:
    def __init__(self):
        # 10 rows x 10 columns
        # blank space = 0
        # agent 0 -- 1
        # agent 1 -- 2
        # flat representation in row major order

        # Main board
        self.squares = np.zeros(
            100,
        )

        # Track player territory (values of 1 or 2 indicate player_0 or player_1)
        self.territory = self.squares.copy()
        self.territory[self.territory > 0] = 0

        self.possible_agents = ["player_0", "player_1"]

        # Piece objects
        self.pieces = {self.possible_agents[i]: get_pieces(i) for i in range(2)}
        self.piece_names = [
            self.pieces["player_0"][piece].label
            for piece in range(len(self.pieces["player_0"]))
        ]
        self.num_pieces = len(self.piece_names)
        self.total_piece_squares = sum(
            [piece.size for piece in self.pieces[self.possible_agents[1]]]
        )
        self.CATHEDRAL_INDEX = self.num_pieces - 1

        # Keep track of which pieces are played (includes cathedral for player_0)
        self.unplaced_pieces = {
            agent: list(np.arange(len(self.pieces[agent])))
            for agent in self.possible_agents
        }

        # Calculate all possible actions and their corresponding positions
        # self.points[agent][piece][action_num] = [(x1, y1), ..., (x5, y5)] for coords (x1,y1), ... (x5, y5)
        self.points = {}
        # self.positions[agent][piece][action_num] = (x, y) for coordinates 0 <= x, y < 10
        self.positions = {}
        # self.rotations[agent][piece][action_num] = 0, 90, 180, or 270
        self.rotations = {}
        # For reverse map function: agent, piece, pos, rotation -> action
        self.reverse_actions = {}
        for agent in self.possible_agents:
            (
                self.points[agent],
                self.positions[agent],
                self.rotations[agent],
                self.reverse_actions[agent],
            ) = self.calculate_possible_actions(agent)

        # Get the total number of actions involving a given piece, using the pre-calculated points dict
        self.num_actions_per_piece = [
            len(self.points["player_0"][piece])
            for piece in self.points["player_0"].keys()
        ]
        # Calculates indices for each piece (for action to piece mapping). Shape: [num_pieces,]
        self.piece_indices = [
            sum(self.num_actions_per_piece[:i])
            for i in range(1, len(self.num_actions_per_piece))
        ]
        self.num_actions = sum(self.num_actions_per_piece)

    def calculate_possible_actions(self, agent):
        points = {}
        positions = {}
        rotations = {}
        reverse_actions = []
        for piece in range(len(self.pieces[agent])):
            points[piece] = []
            positions[piece] = []
            rotations[piece] = []
            for i in range(10):
                for j in range(10):
                    for k in range(4):
                        self.pieces[agent][piece].set_position(i, j)
                        self.pieces[agent][piece].set_rotation(90 * k)
                        if np.all(
                            (np.array(self.pieces[agent][piece].points) >= 0)
                            & (np.array(self.pieces[agent][piece].points) < 10)
                        ):
                            if (
                                set(self.pieces[agent][piece].points)
                                not in points[piece]
                            ):
                                points[piece].append(
                                    set(self.pieces[agent][piece].points)
                                )
                                positions[piece].append(
                                    self.pieces[agent][piece].position
                                )
                                rotations[piece].append(90 * k)
                                reverse_actions.append(
                                    np.array(
                                        (
                                            piece,
                                            self.pieces[agent][piece].position[0],
                                            self.pieces[agent][piece].position[1],
                                            90 * k,
                                        )
                                    )
                                )

            # Mark this piece as unplaced (ref point -1, -1)
            self.pieces[agent][piece].set_unplaced()

            # Reset rotation for consistency
            self.pieces[agent][piece].set_rotation(0)

        return points, positions, rotations, np.array(reverse_actions)

    def check_territory(self, agent):
        self.previous_territory = self.territory.copy()
        piece_removed_size = 0

        # Check if this move creates territory and results in an opponent's piece being removed
        opponent = self.possible_agents[1 - self.possible_agents.index(agent)]
        squares_real = self.squares.copy()
        placed_pieces = [
            (piece, opponent, i)
            for i, piece in enumerate(self.pieces[opponent])
            if piece.is_placed()
        ]
        if agent == "player_0" and self.pieces["player_0"][14].is_placed():
            placed_pieces.append((self.pieces[agent][14], agent, 14))

        # Look through opponent pieces (and the cathedral) and try removing them
        for (piece, piece_agent, piece_idx) in placed_pieces:
            self.squares = squares_real.copy()  # Reset self.squares to original state
            for coord in piece.points:
                self.squares.reshape(10, 10)[coord[0], coord[1]] = 0

            self.get_territory()
            agent_number = self.possible_agents.index(agent) + 1

            if self.territory.max() > 0:
                agent_territory = np.array(
                    np.where(self.territory.reshape(10, 10) == agent_number)
                ).T
                # Convert to list of tuples
                agent_territory = {(coord[0], coord[1]) for coord in agent_territory}

                # If the opponent's piece is now fully within our territory, remove it from the board
                if len(set(piece.points).intersection(agent_territory)) > 0:
                    self.squares = squares_real.copy()  # Reset squares

                    # Get the size of the piece to be removed
                    piece_removed_size = piece.size
                    # Remove this piece fully (mark as unplaced, reset pos/rotation)
                    self.remove(piece_agent, piece_idx)
                    # Update the real squares to reflect this piece being removed
                    squares_real = self.squares.copy()

            self.squares = squares_real  # Reset the squares to the correct state (potentially with the piece removed)
        territory_claimed = (
            self.get_territory()
        )  # Recalculate territory with the proper pieces

        return territory_claimed, piece_removed_size

    def get_territory(self):
        # Reset illegal territory from previous calculations
        self.territory[self.territory < 0] = 0
        self.empty_spaces = [
            (i, j)
            for i in range(10)
            for j in range(10)
            if self.squares.reshape(10, 10)[i, j] == 0
        ]
        while self.empty_spaces:
            # pick a random empty place
            empty_space_seed = next(iter(self.empty_spaces))
            self.remove_empty_spaces(empty_space_seed)

        territory_claimed = len(self.territory[self.territory > 0]) - len(
            self.previous_territory[self.previous_territory > 0]
        )
        return territory_claimed

    def remove_empty_spaces(self, coordinates):
        queue = [coordinates]
        bordering_pieces = {1: [], 2: [], 3: []}
        territory = []
        return_val = None
        visited = []

        while queue:
            x, y = queue.pop()
            try:  # Check if the space is empty
                self.empty_spaces.remove((x, y))
                visited.append((x, y))
                territory.append((x, y))

                new_spaces = [
                    (x - 1, y),
                    (x + 1, y),
                    (x - 1, y + 1),
                    (x + 1, y + 1),
                    (x, y + 1),
                    (x - 1, y - 1),
                    (x + 1, y - 1),
                    (x, y - 1),
                ]
                # Clear illegal and visited indices
                new_spaces = [
                    coord
                    for coord in new_spaces
                    if 0 <= coord[0] < 10
                    and 0 <= coord[1] < 10
                    and coord not in visited
                    and coord not in queue
                ]
                queue += new_spaces

            except ValueError:  # If the space is not empty
                try:
                    # visited.append((x, y)) #
                    # If the current territory is adjacent to something which we already found to be invalid territory
                    if self.territory.reshape(10, 10)[(x, y)] == -1:
                        return_val = -1
                        break
                    # If the current territory is adjacent to an already placed piece
                    if self.squares.reshape(10, 10)[(x, y)] != 0:
                        # If we have already visited this square, skip it # TODO: find edge cases where this actually happens
                        if (x, y) not in bordering_pieces[
                            self.squares.reshape(10, 10)[(x, y)]
                        ]:
                            # Mark the color of the piece
                            bordering_pieces[
                                self.squares.reshape(10, 10)[(x, y)]
                            ].append((x, y))

                            # If it is bordered by more than one piece from each team, then it is not territory
                            if (
                                len(bordering_pieces[1]) > 0
                                and len(bordering_pieces[2]) > 0
                            ):
                                return_val = -1
                                break
                            # If it is bordered by one team's pieces and the cathedral, then it is not territory
                            if (
                                len(bordering_pieces[1]) > 0
                                or len(bordering_pieces[2]) > 0
                            ) and len(bordering_pieces[3]) > 0:
                                return_val = -1
                                break

                except KeyError:  # If x or y is out of range
                    continue
                except IndexError:
                    continue

        if return_val is None:
            # If this "territory" is bordered with both team's pieces, then it is not territory
            # Anything connected to this 'multi team territory' will also be non-territory
            # If it is bordered by more than one piece from each team, then it is not territory
            if len(bordering_pieces[1]) > 0 and len(bordering_pieces[2]) > 0:
                return_val = -1

            # If it is bordered by both teams pieces and the cathedral, then it is not territory
            elif (len(bordering_pieces[1]) > 0 or len(bordering_pieces[2]) > 0) and len(
                bordering_pieces[3]
            ) > 0:
                return_val = -1

            # If this "territory" is larger than half of the board, it is too big
            # However, it is possible for this large territory to be closed later, so don't mark it as invalid
            elif len(territory) > 10 * 4:
                return_val = -1

            # If this territory only borders pieces from a single team, it is valid
            elif len(bordering_pieces[1]) > 0:
                return_val = 1
            elif len(bordering_pieces[2]) > 0:
                return_val = 2

        for coord in territory:
            self.territory.reshape(10, 10)[coord] = return_val
        return return_val

    # Maps an action to its corresponding piece
    def action_to_piece_map(self, action):
        # Bin the action according to number of possible actions for each piece (e.g., first 100 actions are piece 0)
        piece = np.digitize(action, self.piece_indices)

        # Get the position of the action in the bin (e.g., action 199 is position 99 in bin 1)
        action_num = action - self.piece_indices[piece - 1] if piece > 0 else action

        return piece, action_num

    # Helper function for debugging
    def action_to_pos_rotation_mapp(self, agent, action):
        piece, action_num = self.action_to_piece_map(action)
        pos = self.positions[agent][piece][action_num]
        rotation = self.rotations[agent][piece][action_num]
        return pos, rotation

    # Used by manual policy: allows user to select an action visually
    def reverse_action_map(self, agent, piece, pos, rotation):
        try:
            return np.where(
                (self.reverse_actions[agent].T[0] == piece)
                * (self.reverse_actions[agent].T[1] == pos[0])
                * (self.reverse_actions[agent].T[2] == pos[1])
                * (self.reverse_actions[agent].T[3] == rotation)
            )[0][0]
        except IndexError:
            return -1

    def is_legal(self, agent, action):
        piece, action_num = self.action_to_piece_map(action)

        # If the cathedral has not been played, make all other moves illegal
        if self.CATHEDRAL_INDEX in self.unplaced_pieces[agent]:
            if piece != self.CATHEDRAL_INDEX:
                return False

        # If a piece has already been placed, mark it as an illegal moves
        if piece not in self.unplaced_pieces[agent]:
            return False

        # Get the coordinate points which this piece occupies
        points = self.points[agent][piece][action_num]

        agent_idx = self.possible_agents.index(agent)  # player_1 -> 0, player_2 -> 1
        opponent_idx = 1 - agent_idx

        for coord in points:
            # If the spot is occupied by a player's piece or the cathedral, this move is illegal
            if self.squares.reshape(10, 10)[coord[0], coord[1]] in [1, 2, 3]:
                return False
            # Check if territory belongs to other player (player_1 territory: 1, player_2 territory: 2)
            if self.territory.reshape(10, 10)[coord[0], coord[1]] == opponent_idx + 1:
                return False
        return True

    def play_turn(self, agent, action):
        piece_idx, action_num = self.action_to_piece_map(action)
        rotation = self.rotations[agent][piece_idx][action_num]

        # Set this piece as placed (remove from list of unplaced pieces)
        self.unplaced_pieces[agent].remove(piece_idx)

        # Get pre-calculated points for a given piece and action number
        points = self.points[agent][piece_idx][action_num]
        position = self.positions[agent][piece_idx][action_num]

        # Update the piece object's position (we can use this to access the points it occupies)
        piece = self.pieces[agent][piece_idx]

        # Split tuple (x, y) into inputs x, y
        piece.set_position(position[0], position[1])
        piece.set_rotation(rotation)
        piece.set_placed()

        for coord in points:
            if piece_idx == self.CATHEDRAL_INDEX:
                # Cathedral is neither team's piece
                self.squares.reshape(10, 10)[coord[0], coord[1]] = 3
            elif self.possible_agents.index(agent) == 0:
                self.squares.reshape(10, 10)[coord[0], coord[1]] = 1
            elif self.possible_agents.index(agent) == 1:
                self.squares.reshape(10, 10)[coord[0], coord[1]] = 2
        return piece.size

    def preview_turn(self, agent, action):
        piece_idx, action_num = self.action_to_piece_map(action)

        # Get pre-calculated points for a given piece and action number
        points = self.points[agent][piece_idx][action_num]

        for coord in points:
            if piece_idx == self.CATHEDRAL_INDEX:
                self.squares.reshape(10, 10)[coord[0], coord[1]] = (
                    3 + 6
                )  # Cathedral is neither team's piece
            elif self.possible_agents.index(agent) == 0:
                self.squares.reshape(10, 10)[coord[0], coord[1]] = 1 + 6
            elif self.possible_agents.index(agent) == 1:
                self.squares.reshape(10, 10)[coord[0], coord[1]] = 2 + 6
        return

    # Clear any previous previews
    def clear_previews(self):
        self.squares[self.squares > 6] = 0

    def remove(self, agent, piece_idx):
        piece = self.pieces[agent][piece_idx]

        # If the cathedral is removed, it stays removed
        if piece_idx != 14:
            # Mark piece as unplaced
            if piece_idx in self.unplaced_pieces:
                raise Exception(
                    "Trying to remove a piece which is already in the list of unplaced pieces"
                )
            self.unplaced_pieces[agent].append(piece_idx)

        # Mark positions on board as empty
        for coord in piece.points:
            self.squares.reshape(10, 10)[coord[0], coord[1]] = 0

        # Reset piece position
        piece.set_unplaced()
        piece.set_rotation(0)

    # returns:
    # -1 for no winner
    # 0 -- agent 0 wins
    # 1 -- agent 1 wins
    def check_for_winner(self):
        pieces_remaining, piece_score = self.get_score()

        # Lowest total size of remaining pieces wins
        if piece_score[self.possible_agents[0]] < piece_score[self.possible_agents[1]]:
            winner = 0
        elif (
            piece_score[self.possible_agents[0]] > piece_score[self.possible_agents[1]]
        ):
            winner = 1
        else:
            winner = -1
        return winner, pieces_remaining, piece_score

    # Returns list of pieces remaining and total number of points occupied by those pieces (piece score)
    def get_score(self):
        piece_score = {agent: 0 for agent in self.possible_agents}
        pieces_remaining = {agent: [] for agent in self.possible_agents}
        for agent in self.possible_agents:
            for piece_idx in self.unplaced_pieces[agent]:
                piece_score[agent] += self.pieces[agent][piece_idx].size
                pieces_remaining[agent].append(self.pieces[agent][piece_idx])
        return pieces_remaining, piece_score

    def __str__(self):
        return str(self.squares.reshape(10, 10).T)  # Lines up with pygame rendering
