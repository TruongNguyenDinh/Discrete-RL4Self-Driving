import os

from game.env import CarEnv
from rl.ddqn import DDQNAgent


# =====================================================
# Config
# =====================================================

EPISODES = 3000

MODEL_DIR = "./model"

CHECKPOINT_INTERVAL = 100

RENDER = False


# =====================================================
# Train
# =====================================================

def train():

    os.makedirs(MODEL_DIR, exist_ok=True)

    env = CarEnv(render_mode=False)

    state_size = len(env.reset())
    action_size = env.action_size

    agent = DDQNAgent(
        state_size=state_size,
        action_size=action_size
    )

    best_score = -1

    for episode in range(1, EPISODES + 1):

        state = env.reset()

        done = False

        total_reward = 0

        while not done:

            if RENDER:
                env.render()

            # -------------------------
            # Chọn action
            # -------------------------

            action = agent.select_action(state)

            # -------------------------
            # Tương tác với môi trường
            # -------------------------

            next_state, reward, done = env.step(action)

            # -------------------------
            # Lưu replay
            # -------------------------

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            # -------------------------
            # Huấn luyện
            # -------------------------

            agent.train_step()

            state = next_state

            total_reward += reward

        # =====================================================
        # CHANGED: decay epsilon once per episode (not per step).
        # Must be called here -- decay_epsilon() no longer runs
        # automatically inside train_step().
        # =====================================================

        agent.decay_epsilon()

        # =====================================================
        # Save Best Model
        # =====================================================

        if env.score > best_score:

            best_score = env.score

            agent.save(
                os.path.join(
                    MODEL_DIR,
                    "best_model.pth"
                )
            )

            print(f"🏆 New Best Model | Score = {best_score}")

        # =====================================================
        # Save Checkpoint
        # =====================================================

        if episode % CHECKPOINT_INTERVAL == 0:

            agent.save(
                os.path.join(
                    MODEL_DIR,
                    f"model_weights.pth"
                )
            )

            print(f"💾 Saved checkpoint: model_weights.pth")

        # =====================================================
        # Log
        # =====================================================

        print(
            f"Episode: {episode:4d}/{EPISODES}"
            f" | Score: {env.score:4d}"
            f" | Reward: {total_reward:8.2f}"
            f" | Epsilon: {agent.epsilon:.4f}"
        )
    env.close()


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    train()