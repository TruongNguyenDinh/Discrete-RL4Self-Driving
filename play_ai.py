from game.env import CarEnv
from rl.ddqn import DDQNAgent


MODEL_PATH = "./model/best_model.pth"


def main():

    env = CarEnv(render_mode=True)

    state_size = len(env.reset())
    action_size = env.action_size

    agent = DDQNAgent(
        state_size=state_size,
        action_size=action_size
    )

    agent.load(MODEL_PATH)

    # Không khám phá nữa
    agent.epsilon = 0.0

    state = env.reset()

    while True:

        env.render()

        action = agent.select_action(state)

        next_state, reward, done = env.step(action)

        state = next_state

        if done:

            print(f"Game Over | Score = {env.score}")

            state = env.reset()


if __name__ == "__main__":

    main()