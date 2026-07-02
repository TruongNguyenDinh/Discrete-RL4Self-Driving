import random
from .replay_buffer import ReplayBuffer

import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================================
# Q Network
# ==========================================================

class DDQN(nn.Module):

    def __init__(
        self,
        state_size,
        action_size
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)

        )

    def forward(self, x):

        return self.network(x)


# ==========================================================
# DDQN Agent
# ==========================================================

class DDQNAgent:

    def __init__(
        self,
        state_size,
        action_size,
        device=None,
        use_soft_update=True
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ---------------------------
        # Hyper Parameters
        # ---------------------------

        self.gamma = 0.99

        self.learning_rate = 1e-3

        self.batch_size = 64

        self.memory = ReplayBuffer(
            capacity=50000
        )

        # ---------------------------
        # Epsilon Greedy
        # ---------------------------
        # CHANGED: epsilon_min lowered from 0.1 -> 0.03.
        # 0.1 means even a fully-trained policy still takes a random
        # action 10% of the time -> in a lane-changing task this alone
        # is enough to cause visible "unnecessary lane changes" that
        # have nothing to do with what the policy actually learned.
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # CHANGED: decay is now driven by episodes, not by every
        # train_step() call. Decaying per train_step makes the decay
        # speed depend on how many gradient updates happen per
        # episode (i.e. episode length), which is an accident of the
        # environment, not a real training schedule. Call
        # agent.decay_epsilon() once at the end of each episode from
        # your training loop.

        # ---------------------------
        # Warm-up before learning starts
        # ---------------------------
        # CHANGED: added. Learning from a replay buffer that only has
        # batch_size (64) samples means the first hundreds of updates
        # are trained on a very narrow, non-i.i.d. slice of experience,
        # which can push the policy toward a jittery/inconsistent
        # early behavior that later exploration never fully undoes.
        self.warmup_steps = 1000

        # ---------------------------
        # Target Network Update
        # ---------------------------
        # CHANGED: switched default to a soft (Polyak) update instead
        # of a hard copy every 1000 steps. Hard copies cause a sudden
        # jump in the bootstrap target every target_update steps,
        # which shows up as short bursts of unstable/inconsistent
        # Q-values right after each copy -> can look like erratic
        # lane-change decisions. Soft update moves the target network
        # a little bit every step instead, which is much smoother.
        self.use_soft_update = use_soft_update
        self.tau = 0.005          # soft update rate (used if use_soft_update=True)
        self.target_update = 1000  # hard update period (used if use_soft_update=False)
        self.learn_step = 0

        # ---------------------------
        # Networks
        # ---------------------------

        self.policy_net = DDQN(
            state_size,
            action_size
        ).to(self.device)

        self.target_net = DDQN(
            state_size,
            action_size
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )

        self.criterion = nn.SmoothL1Loss()

    # =====================================================

    def select_action(self, state, evaluate=False):
        """
        CHANGED: added `evaluate` flag.
        When evaluate=True, epsilon is ignored entirely (pure greedy
        action). Use this when testing / deploying the trained agent
        so it never takes a random exploratory action -> removes one
        whole source of "unnecessary lane changes" outside of training.
        """

        if (not evaluate) and random.random() < self.epsilon:

            return random.randrange(self.action_size)

        state = torch.FloatTensor(
            state
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            q_values = self.policy_net(state)

        return q_values.argmax(dim=1).item()

    # =====================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    # =====================================================

    def train_step(self):

        # CHANGED: wait for warmup_steps (not just batch_size) samples
        # before starting to learn.
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return

        states, actions, rewards, next_states, dones = \
        self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)

        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)

        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

        next_states = torch.FloatTensor(next_states).to(self.device)

        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ---------------------------
        # Current Q
        # ---------------------------

        current_q = self.policy_net(states).gather(
            1,
            actions
        )

        # ---------------------------
        # Double DQN
        # ---------------------------

        next_actions = self.policy_net(
            next_states
        ).argmax(
            dim=1,
            keepdim=True
        )

        next_q = self.target_net(
            next_states
        ).gather(
            1,
            next_actions
        )

        target_q = rewards + (
            1 - dones
        ) * self.gamma * next_q

        # ---------------------------
        # Backpropagation
        # ---------------------------

        loss = self.criterion(
            current_q,
            target_q.detach()
        )

        self.optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0
        )
        self.optimizer.step()

        # ---------------------------
        # Update Target Network
        # ---------------------------

        self.learn_step += 1

        if self.use_soft_update:
            # CHANGED: Polyak averaging, applied every step.
            with torch.no_grad():
                for target_param, policy_param in zip(
                    self.target_net.parameters(),
                    self.policy_net.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * policy_param.data
                        + (1.0 - self.tau) * target_param.data
                    )
        else:
            if self.learn_step % self.target_update == 0:
                self.target_net.load_state_dict(
                    self.policy_net.state_dict()
                )

        # NOTE: epsilon decay removed from here.
        # Call self.decay_epsilon() once per episode instead.

    # =====================================================

    def decay_epsilon(self):
        """
        CHANGED: new method. Call this once at the end of every
        episode (not every train_step) so exploration decays on a
        schedule tied to episodes, which is what you actually control
        and reason about when tuning.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    # =====================================================

    def save(self, path):

        torch.save(
            self.policy_net.state_dict(),
            path
        )

    # =====================================================

    def load(self, path):

        self.policy_net.load_state_dict(
            torch.load(
                path,
                map_location=self.device
            )
        )

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.policy_net.eval()
        self.target_net.eval()