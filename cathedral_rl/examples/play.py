import argparse

import numpy as np

from cathedral_rl import cathedral_v0
from cathedral_rl.game.manual_policy import ManualPolicy


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array", "text", "text_full"],
        help="Choose the rendering mode for the game.",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for board and policy"
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = get_args()

    env = cathedral_v0.env(render_mode=args.render_mode)

    env.reset(args.seed)
    env.render()

    env.action_space("player_0").seed(args.seed)
    env.action_space("player_1").seed(args.seed)

    iter = 1

    manual_policy = ManualPolicy(env)
    manual_policy.agent = "player_1"

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        mask = observation["action_mask"]

        print(
            f"\nTurn: {iter} | Player: {agent}, Number of legal moves: ",
            np.count_nonzero(mask),
        )

        if termination or truncation:
            print("Terminated") if termination else print("Truncated")
            print("\nWINNER: ", env.unwrapped.winner)
            for agent in env.possible_agents:
                print(f"\n{agent} Final reward: ", env.unwrapped.rewards[agent])
                print(f"{agent} Final score: ", env.unwrapped.score[agent])
                print(
                    f"{agent} Final pieces left over: ",
                    [p.name for p in env.unwrapped.final_pieces[agent]],
                )
            env.step(None)
            break

        if agent == manual_policy.agent:
            action = manual_policy(observation, agent)
        else:
            action = env.action_space(agent).sample(mask=mask)

        env.step(action)

        print(
            f"Turn: {iter} | Action: {action}, Piece: {env.unwrapped.board.action_to_piece_map(action)[0]}, "
            f"Position: {env.unwrapped.board.action_to_pos_rotation_mapp(agent, action)[0]} Remaining pieces: {env.unwrapped.board.unplaced_pieces[agent]}"
        )

        iter += 1
