# noqa
"""
# Connect Four

```{figure} classic_connect_four.gif
:width: 140px
:name: connect_four
```

This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.classic import connect_four_v3` |
|--------------------|--------------------------------------------------|
| Actions            | Discrete                                         |
| Parallel API       | Yes                                              |
| Manual Control     | Yes                                              |
| Agents             | `agents= ['player_0', 'player_1']`               |
| Agents             | 2                                                |
| Action Shape       | (1,)                                             |
| Action Values      | Discrete(7)                                      |
| Observation Shape  | (6, 7, 2)                                        |
| Observation Values | [0,1]                                            |


Connect Four is a 2-player turn based game, where players must connect four of their tokens vertically, horizontally or diagonally. The players drop their respective token in a column of a standing grid, where each token will fall until it reaches the bottom of the column or reaches an existing
token. Players cannot place a token in a full column, and the game ends when either a player has made a sequence of 4 tokens, or when all 7 columns have been filled.

### Observation Space

The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.


The main observation space is 5 channels each representing planes of a 10x10 grid:
 * Channel 1: White pieces: a 1 indicates a white piece occupies a given space, 0 means either the space is empty, or the opponent has a piece occupying that space.
 * Channel 2: Black pieces: a 1 indicates a brown piece occupies a given space, 0 means either the space is empty, or the opponent has a piece occupying that space.
 * Channel 3: Cathedral: a 1 indicates the cathedral occupies a given space, 0 means that space is empty or is occupied by a player's piece.
 * Channel 4: White territory: a 1 indicates a given space is part of white's territory, a 0 means the space is occupied, and -1 indicates it is not part of white's territory
 * Channel 5: Black territory: a 1 indicates a given space is part of white's territory, a 0 means the space is occupied, and -1 indicates it is not part of brown's territory

The first and second panes represent a specific agent's tokens, and each location in the grid represents the placement of the corresponding agent's token. 1 indicates that the agent has a token placed in that cell, and 0 indicates they do not have a token in
that cell. A 0 means that either the cell is empty, or the other agent has a token in that cell.



#### Legal Actions Mask

The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation.
The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.


### Action Space

The action space is the set of integers from 0 to 6 (inclusive), where the action represents which column a token should be dropped in.

### Rewards

If an agent successfully connects four of their tokens, they will be rewarded 1 point. At the same time, the opponent agent will be awarded -1 points. If the game ends in a draw, both players are rewarded 0.


### Version History

* v0: Initial versions release (1.0.0)

"""

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from .board import Board


def env(render_mode=None):
    env = raw_env(render_mode=render_mode)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "cathedral_v0",
        "is_parallelizable": False,
        "render_fps": 0,
        "has_manual_policy": True,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.screen = None
        self.render_mode = render_mode

        # Pygame setup
        if render_mode == "human":
            pygame.init()
            self.clock = pygame.time.Clock
            self.WINDOW_WIDTH = 1000
            self.WINDOW_HEIGHT = 1000
            self.window = pygame.display.set_mode(
                (self.WINDOW_HEIGHT, self.WINDOW_HEIGHT)
            )
            self.clock = pygame.time.Clock()
            self.WINDOW_WIDTH, self.WINDOW_HEIGHT = self.window.get_size()

        self.board = Board()

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {
            i: spaces.Discrete(self.board.num_actions) for i in self.agents
        }
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(10, 10, 5), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.board.num_actions,), dtype=np.int8
                    ),
                }
            )
            for i in self.agents
        }

    # Key
    # ----
    # blank space = 0
    # agent 0 = 1
    # agent 1 = 2
    # An observation is list of lists, where each list represents a row
    #
    # array([[0, 1, 1, 2, 0, 1, 0],
    #        [1, 0, 1, 2, 2, 2, 1],
    #        [0, 1, 0, 0, 1, 2, 1],
    #        [1, 0, 2, 0, 1, 1, 0],
    #        [2, 0, 0, 0, 1, 1, 0],
    #        [1, 1, 2, 1, 0, 1, 0]], dtype=int8)
    def observe(self, agent):
        board_vals = self.board.squares.reshape(10, 10)
        board_territory = self.board.territory.reshape(10, 10)
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        cur_p_board = np.equal(board_vals, cur_player + 1)
        opp_p_board = np.equal(board_vals, opp_player + 1)
        cathedral_board = np.equal(board_vals, 3)
        cur_p_territory = np.equal(board_territory, cur_player + 1)
        opp_p_territory = np.equal(board_territory, opp_player + 1)

        layers = [
            cur_p_board,
            opp_p_board,
            cathedral_board,
            cur_p_territory,
            opp_p_territory,
        ]

        observation = np.stack(layers, axis=2).astype(np.int8)
        legal_moves = self._legal_moves(agent) if agent == self.agent_selection else []

        action_mask = np.zeros(self.action_space(agent).n, dtype=np.int8)
        for i in range(self.action_space(agent).n):
            if i in legal_moves:
                action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _legal_moves(self, agent):
        legal_moves = []
        for i in range(self.board.num_actions):
            if self.board.is_legal(agent, i):
                legal_moves.append(i)

        return legal_moves

    def _calculate_winner(self):
        winner, pieces_remaining, score = self.board.check_for_winner()
        if winner == 0:
            self.rewards[self.agents[0]] = 1
            self.rewards[self.agents[1]] = -1
        elif winner == 1:
            self.rewards[self.agents[0]] = -1
            self.rewards[self.agents[1]] = 1
        else:
            self.rewards[self.agents[0]] = 0
            self.rewards[self.agents[1]] = 0

        self.winner = winner
        self.score = score
        self.final_pieces = pieces_remaining
        self.terminations = {i: True for i in self.agents}

    def step(self, action):
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        # check that it is a valid move
        if not self.board.is_legal(self.agent_selection, action):
            raise Exception("played illegal move.")

        self.board.play_turn(self.agent_selection, action)
        self.board.check_territory(self.agent_selection)

        next_agent = self._agent_selector.next()

        # Only switch agents if the next agent has legal moves
        # Otherwise, the current agent keeps placing pieces
        if len(self._legal_moves(next_agent)) != 0:
            self.agent_selection = next_agent

        elif len(self._legal_moves(self.agent_selection)) == 0:
            self._calculate_winner()

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, return_info=False, options=None):
        # reset environment
        self.board = Board()

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        # Additional info about game outcome
        self.winner = -1
        self.score = {agent: 0 for agent in self.agents}
        self.final_pieces = {agent: [] for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)

        self.agent_selection = self._agent_selector.reset()

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        elif self.render_mode == "human":
            self.clock.tick(self.metadata["render_fps"])

            # Only render if there is something to render
            block_size = int(self.WINDOW_WIDTH / 10)  # Set the size of the grid block
            border_color = (0, 0, 0)
            self.colors = {
                0: (211, 211, 211),
                1: (221, 186, 151),
                2: (120, 65, 65),
                3: (128, 128, 162),
                4: (238, 221, 203),
                5: (188, 160, 160),
                6: (0, 0, 0),
                7: (233, 210, 187),
                8: (208, 189, 189),
                9: (172, 172, 195),
            }  # old cathedral preview color (192,221,208)

            for x, x_screen in enumerate(range(0, self.WINDOW_WIDTH, block_size)):
                for y, y_screen in enumerate(range(0, self.WINDOW_HEIGHT, block_size)):
                    # If space is empty and in a player's territory, color it to mark that it is territory
                    if (
                        self.board.territory.reshape(10, 10)[x, y] > 0
                        and self.board.squares.reshape(10, 10)[x, y] == 0
                    ):
                        color = self.colors[
                            self.board.territory.reshape(10, 10)[x, y] + 3
                        ]
                    else:
                        color = self.colors[self.board.squares.reshape(10, 10)[x, y]]

                    if color == self.colors[0]:
                        border = False
                    else:
                        border = True

                    def draw_square(
                        surface, x, y, width, height, color, border_color, border
                    ):
                        pygame.draw.rect(surface, color, (x, y, width, height), 0)
                        if border:
                            pygame.draw.rect(
                                surface, border_color, (x, y, width, height), 2
                            )
                        else:
                            pygame.draw.rect(
                                surface, border_color, (x, y, width, height), 1
                            )

                    draw_square(
                        self.window,
                        x_screen,
                        y_screen,
                        block_size,
                        block_size,
                        color,
                        border_color,
                        border,
                    )

            pygame.display.update()
        elif self.render_mode == "text":
            print("Board: \n", self.board.squares.reshape(10, 10))
            print("Territory: \n", self.board.territory.reshape(10, 10))

    def close(self):
        if self.render_mode == "human":
            import pygame

            pygame.quit()
            self.screen = None
