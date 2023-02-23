import sys

import numpy as np
import pygame

from .utils import GIFRecorder


class ManualPolicy:
    def __init__(self, env, agent_id: int = 0, recorder: GIFRecorder = None):

        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]
        self.recorder = recorder

    def __call__(self, observation, agent):
        # only trigger when we are the correct agent
        assert (
            agent == self.agent
        ), f"Manual Policy only applied to agent: {self.agent}, but got tag for {agent}."

        piece_cycle = 0
        selected_piece = -1
        rotation = 0
        pos = (4, 4)  # Default position in center of board
        mousex, mousey = (
            0,
            0,
        )  # set first mouse coordinates at (0, 0) as that's the default with cursor offscreen
        action = -1

        # set the default action
        while True:
            event = pygame.event.wait()

            recorder = self.recorder
            env = self.env

            if event.type == pygame.QUIT:
                if recorder is not None:
                    recorder.end_recording(env.unwrapped.screen)
                pygame.quit()
                pygame.display.quit()
                sys.exit()

            """ GET MOUSE INPUT"""
            if pygame.mouse.get_focused():  # If the cursor is hovering over the screen
                mousex_prev, mousey_prev = mousex, mousey
                mousex, mousey = pygame.mouse.get_pos()

                if mousex != mousex_prev or mousey != mousey_prev:
                    bins = np.arange(
                        0, 1000, 100
                    )  # Ten bins (100, ... 1000), offset by 1 to be in range [0, 10)
                    pos = (np.digitize(mousex, bins) - 1, np.digitize(mousey, bins) - 1)

                    # print(f"mousex: {mousex}, mousey: {mousey}")
                    # print(f"x bin: {pos[0]}, y bin: {pos[1]}")
                # else:
                # print("AFK mouse")

            # print("piece selected: ", selected_piece)

            """ FIND PLACED PIECES """
            unplaced = env.unwrapped.board.unplaced_pieces[agent]

            # Default piece choice: last piece in the cycle (largest piece)
            if selected_piece == -1:
                if len(unplaced) > 0:
                    selected_piece = unplaced[len(unplaced) - 1]
                else:
                    selected_piece = -1

            """ READ KEYBOARD INPUT"""
            if event.type == pygame.KEYDOWN:
                if (
                    event.key == pygame.K_SPACE
                ):  # Cycle through pieces (from largest to smallest)
                    piece_cycle = (piece_cycle + 1) % len(unplaced)
                    if len(unplaced) > 0:
                        selected_piece = unplaced[piece_cycle]
                elif (
                    event.key == pygame.K_e
                ):  # E key: rotate piece clockwise (flipped because of flipped board)
                    rotations = 0
                    while rotations < 4:
                        rotation = (rotation - 90) % 360
                        rotations += 1
                        act = env.unwrapped.board.reverse_action_map(
                            agent, selected_piece, pos, rotation
                        )
                        if act != -1:
                            break
                elif event.key == pygame.K_q:  # Q key: rotate piece counter-clockwise
                    rotations = 0
                    while rotations < 4:
                        rotation = (rotation + 90) % 360
                        rotations += 1
                        act = env.unwrapped.board.reverse_action_map(
                            agent, selected_piece, pos, rotation
                        )
                        if act != -1:
                            break

                elif event.key == pygame.K_RIGHT:  # Right arrow: move piece right
                    pos_test = pos
                    while pos_test[0] < 10:
                        pos_test = (pos_test[0] + 1, pos_test[1])
                        act = env.unwrapped.board.reverse_action_map(
                            agent, selected_piece, pos_test, rotation
                        )
                        if act != -1:
                            if env.unwrapped.board.is_legal(agent, act):
                                pos = pos_test
                                break
                elif event.key == pygame.K_LEFT:  # Left arrow: move piece left
                    pos_test = pos
                    while pos_test[0] > 0:
                        pos_test = (pos_test[0] - 1, pos_test[1])
                        act = env.unwrapped.board.reverse_action_map(
                            agent, selected_piece, pos_test, rotation
                        )
                        if act != -1:
                            if env.unwrapped.board.is_legal(agent, act):
                                pos = pos_test
                                break
                elif (
                    event.key == pygame.K_UP
                ):  # Up arrow: move piece up (pygame y value starts from top)
                    pos_test = pos
                    while pos_test[1] > 0:
                        pos_test = (pos_test[0], pos_test[1] - 1)
                        act = env.unwrapped.board.reverse_action_map(
                            agent, selected_piece, pos_test, rotation
                        )
                        if act != -1:
                            if env.unwrapped.board.is_legal(agent, act):
                                pos = pos_test
                                break
                elif event.key == pygame.K_DOWN:  # Down arrow: move piece down
                    pos_test = pos
                    while pos_test[1] < 10:
                        pos_test = (pos_test[0], pos_test[1] + 1)
                        act = env.unwrapped.board.reverse_action_map(
                            agent, selected_piece, pos_test, rotation
                        )
                        if act != -1:
                            if env.unwrapped.board.is_legal(agent, act):
                                pos = pos_test
                                break

            """ GET PREVIEW ACTION """
            # Get the action from the preview (in position the mouse cursor is currently hovering over)
            action_prev = env.unwrapped.board.reverse_action_map(
                agent, selected_piece, pos, rotation
            )

            env.unwrapped.board.clear_previews()

            """ CLEAR ACTION PREVIEW FOR ILLEGAL MOVES"""
            if action_prev != -1 and env.unwrapped.board.is_legal(agent, action_prev):
                env.unwrapped.board.preview_turn(agent, action_prev)

                """ UPDATE DISPLAY with previewed move"""
                env.render()
                pygame.display.update()
                if recorder is not None:
                    recorder.capture_frame(env.unwrapped.screen)

                action = (
                    action_prev  # Set action to return as the most recent legal action
                )

            if action != -1:
                """PICK UP / PLACE A PIECE"""
                if event.type == pygame.MOUSEBUTTONDOWN or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN
                ):
                    # Place a piece (if it is legal to do so)
                    env.unwrapped.board.clear_previews()
                    return action

    @property
    def available_agents(self):
        return self.env.agent_name_mapping
