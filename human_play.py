# human_play.py

import pygame
from game.env import CarEnv


def main():

    env = CarEnv()

    state = env.reset()

    running = True

    while running:

        action = 1  # Mặc định: Stay

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_LEFT:
                    action = 0

                elif event.key == pygame.K_RIGHT:
                    action = 2

        state, reward, done = env.step(action)

        env.render()

        pygame.display.set_caption(
            f"Score: {env.score} | Reward: {reward:.1f}"
        )

        if done:
            print("Game Over!")
            print("Final Score:", env.score)
            running = False

    env.close()


if __name__ == "__main__":
    main()