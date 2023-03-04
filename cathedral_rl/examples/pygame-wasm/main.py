import asyncio
import sys
import time

sys.path.append("modules")

import numpy as np  # noqa: E402

from cathedral_rl import cathedral_v0  # noqa: E402
from cathedral_rl.game.manual_policy import ManualPolicy  # noqa: E402


async def main():
    env = cathedral_v0.env(
        render_mode="human", per_move_rewards=True, final_reward_score_difference=True
    ).unwrapped

    seed = None
    env.reset(seed)
    env.render()

    env.action_space("player_0").seed(seed)
    env.action_space("player_1").seed(seed)

    iter = 1

    manual_policy = ManualPolicy(env)

    while env.agents:
        observation, reward, termination, truncation, info = env.last()
        mask = observation["action_mask"]
        agent = env.agent_selection

        print(
            f"\nTurn: {iter} | ({agent}) "
            f"Legal pieces : {list(env.legal_pieces[agent])}, "
            f"Legal moves total: {np.count_nonzero(mask)}, "
            f"Remaining pieces: {env.board.unplaced_pieces[agent]}"
        )

        if agent == manual_policy.agent:
            action = manual_policy(observation, agent)
        else:
            action = env.action_space(agent).sample(mask=mask)
            time.sleep(0.25)

        env.step(action)

        print(
            f"Turn: {iter} | "
            f"Action: {action}, "
            f"Piece: {env.board.action_to_piece_map(action)[0]}, "
            f"Position: {env.board.action_to_pos_rotation_mapp(agent, action)[0]}, "
        )
        print(
            f"Turn: {iter} | Reward: {env.rewards[agent]}, "
            f"Cumulative reward: {env._cumulative_rewards[agent]}, "
        )
        if env.turns["player_0"] == env.turns["player_1"]:
            print()
            for agent in env.agents:
                print(
                    f"SCORE ({agent}): {env.score[agent]['total']:0.2f}, "
                    f"Squares/turn: {env.score[agent]['squares_per_turn']:0.2f}, "
                    f"Remaining pieces difference: {env.score[agent]['remaining_pieces']}, "
                    f"Territory difference: {env.score[agent]['territory']}"
                )

        iter += 1

    print("Terminated") if termination else print("Truncated")
    print("\nWINNER: ", env.winner)
    for agent in env.possible_agents:
        print(f"\n{agent} Final reward: {env.rewards[agent]}")
        print(f"{agent} Cumulative reward: {env._cumulative_rewards[agent]}")
        print(
            f"{agent} Final remaining pieces: {[p.name for p in env.final_pieces[agent]]}"
        )
        print(
            f"{agent} Score: {env.score[agent]['total']:0.2f}, "
            f"Squares/turn: {env.score[agent]['squares_per_turn']:0.2f}, "
            f"Remaining pieces difference: {env.score[agent]['remaining_pieces']}, "
            f"Territory difference: {env.score[agent]['territory']}"
        )

    await asyncio.sleep(0)  # Very important, and keep it 0


if __name__ == "__main__":
    asyncio.run(main())
