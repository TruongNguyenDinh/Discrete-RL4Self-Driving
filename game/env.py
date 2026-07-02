import random
import pygame

from .player_car import PlayerCar
from .enemy_car import EnemyCar


class CarEnv:

    def __init__(self, render_mode=True):
        # Screen
        # ==========================

        self.WIDTH = 600
        self.HEIGHT = 800
        self.render_mode = render_mode

        if self.render_mode:
            pygame.init()

            self.screen = pygame.display.set_mode(
                (self.WIDTH, self.HEIGHT)
            )

            pygame.display.set_caption(
                "RL Self Driving"
            )

            self.clock = pygame.time.Clock()

            self.font = pygame.font.SysFont(
                None,
                36
            )
        else:

            self.screen = None
            self.clock = None
            self.font = None



        if self.render_mode:

            self.screen = pygame.display.set_mode(
                (self.WIDTH, self.HEIGHT)
            )

            pygame.display.set_caption(
                "RL Self Driving"
            )
        else:

            self.screen = None

        # ==========================
        # Config
        # ==========================

        self.FPS = 60

        self.NUM_LANES = 3
        self.NUM_WAVES = 4

        self.WAVE_GAP = 320

        self.LANE_X = [170, 300, 430]

        self.base_speed = 7

        self.action_size = 3

        self.previous_pattern = None

        # ---------------------------
        # CHANGED: reward shaping constants
        # ---------------------------
        # Pulled out as named constants so they're easy to tune, and
        # so the shaping logic below is readable.

        self.SURVIVAL_REWARD = 0.1
        self.WAVE_CLEAR_REWARD = 1.0
        self.COLLISION_PENALTY = -5.0

        # Base cost of changing lane at all (was -0.1, and was exactly
        # cancelling SURVIVAL_REWARD -> net 0, i.e. no real penalty).
        self.LANE_CHANGE_PENALTY = 0.3

        # Extra penalty ONLY when the lane change was not needed
        # (no enemy close in the lane the player was leaving).
        self.UNNECESSARY_CHANGE_PENALTY = 0.2

        # Extra penalty for changing lane again right after a
        # previous change (weaving / oscillation).
        self.OSCILLATION_PENALTY = 0.2
        self.OSCILLATION_WINDOW = 5  # steps

        # Normalized distance (0..1) below which an enemy in the
        # player's current lane counts as "close enough to justify"
        # a lane change.
        self.DANGER_THRESHOLD = 0.28

        # Số wave tách riêng để agent nhìn thấy trước (không gộp chung)
        self.LOOKAHEAD_WAVES = 2

        self.reset()

    # ===================================================

    def generate_pattern(self):

        while True:

            pattern = [0, 0, 0]

            first = random.randint(0, 2)
            pattern[first] = 1

            remain = [i for i in range(3) if i != first]

            second = random.choice(remain)

            if random.random() < 0.5:
                pattern[second] = 1

            if self.previous_pattern is None:

                self.previous_pattern = pattern

                return pattern

            if pattern != self.previous_pattern:

                self.previous_pattern = pattern

                return pattern

    # ===================================================

    def create_wave(self, y):

        pattern = self.generate_pattern()

        enemies = []

        for lane in range(self.NUM_LANES):

            if pattern[lane]:

                enemy = EnemyCar(
                    lane_x=self.LANE_X,
                    y=y,
                    speed=self.base_speed,
                    color=(255, 0, 0)
                )

                enemy.lane = lane
                enemy.update()

                enemies.append(enemy)

        return enemies

    # ===================================================

    def reset(self):

        self.player = PlayerCar(
            lane=1,
            lane_x=self.LANE_X,
            y=650,
            color=(0, 255, 0)
        )

        self.score = 0

        self.previous_pattern = None

        # CHANGED: tracks how many steps since the last lane change,
        # used for the oscillation penalty.
        self.steps_since_lane_change = self.OSCILLATION_WINDOW

        self.waves = []

        y = -150

        for _ in range(self.NUM_WAVES):

            self.waves.append(
                self.create_wave(y)
            )

            y -= self.WAVE_GAP

        return self.get_state()

    # ===================================================

    def get_state(self):

        lane_norm = self.player.lane / (self.NUM_LANES - 1)

        state = [lane_norm]

        per_wave_distances = []

        for wave in self.waves:

            dists = [None] * self.NUM_LANES

            for enemy in wave:

                if enemy.y < self.player.y:

                    d = (self.player.y - enemy.y) / self.HEIGHT

                    if dists[enemy.lane] is None or d < dists[enemy.lane]:
                        dists[enemy.lane] = d

            relevant = [d for d in dists if d is not None]

            if relevant:
                per_wave_distances.append((min(relevant), dists))

        # Wave gần nhất đứng trước
        per_wave_distances.sort(key=lambda item: item[0])

        for i in range(self.LOOKAHEAD_WAVES):

            if i < len(per_wave_distances):
                dists = per_wave_distances[i][1]
            else:
                dists = [None] * self.NUM_LANES

            for lane in range(self.NUM_LANES):
                state.append(dists[lane] if dists[lane] is not None else 1.0)

        return state

    def step(self, action):

        reward = self.SURVIVAL_REWARD
        done = False

        # -------------------------
        # Thực hiện hành động
        # -------------------------

        old_lane = self.player.lane

        # CHANGED: read distances in the CURRENT lane *before* moving,
        # so we can tell whether a lane change was actually justified
        # (an enemy was close) or not.
        pre_state = self.get_state()
        pre_distances = pre_state[1:]
        was_in_danger = pre_distances[old_lane] < self.DANGER_THRESHOLD

        self.player.step(action)
        self.player.update()

        # -------------------------
        # CHANGED: lane-change reward shaping
        # -------------------------
        # Previously: reward -= 0.1, which exactly cancelled the
        # +0.1 survival reward -> a lane change cost nothing relative
        # to standing still, so the agent had no real signal against
        # weaving. Now:
        #   - every lane change costs LANE_CHANGE_PENALTY outright
        #   - changing lane with nothing nearby costs extra
        #     (UNNECESSARY_CHANGE_PENALTY)
        #   - changing lane again shortly after a previous change
        #     costs extra (OSCILLATION_PENALTY), to directly target
        #     rapid back-and-forth weaving.
        # A change made to dodge a close enemy is only charged the
        # base cost, so evasive maneuvers aren't punished as hard as
        # pointless ones.

        if self.player.lane != old_lane:

            reward -= self.LANE_CHANGE_PENALTY

            if not was_in_danger:
                reward -= self.UNNECESSARY_CHANGE_PENALTY

            if self.steps_since_lane_change < self.OSCILLATION_WINDOW:
                reward -= self.OSCILLATION_PENALTY

            self.steps_since_lane_change = 0

        else:

            self.steps_since_lane_change += 1

        # -------------------------
        # Cập nhật wave
        # -------------------------

        # CHANGED: compute the spawn anchor ONCE before the loop.
        # The old code recomputed `highest` from self.waves *inside*
        # the loop, after already overwriting self.waves[i] with a
        # brand-new (very-far-up) wave. If two waves cleared on the
        # same frame, the second new wave would anchor itself off the
        # first new wave's position instead of the real formation,
        # silently breaking wave spacing and occasionally spawning
        # enemies in unpredictable/bunched patterns - which then
        # forces sudden, hard-to-learn evasive lane changes.
        next_spawn_y = min(
            min(e.y for e in w)
            for w in self.waves
        )

        for i, wave in enumerate(self.waves):

            remove_wave = True

            for enemy in wave:

                enemy.update()

                # Va chạm
                if self.player.collide(enemy):

                    reward = self.COLLISION_PENALTY
                    done = True
                    break

                if enemy.y <= self.HEIGHT:
                    remove_wave = False

            if done:
                break

            # Wave đi hết màn hình
            if remove_wave:

                next_spawn_y -= self.WAVE_GAP

                self.waves[i] = self.create_wave(
                    next_spawn_y
                )

                self.score += 1
                reward += self.WAVE_CLEAR_REWARD

        next_state = self.get_state()

        return next_state, reward, done

    def render(self):
        if not self.render_mode:
            return
        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                self.close()

                return

        self.screen.fill((60, 60, 60))

        for x in self.LANE_X:

            for y in range(0, self.HEIGHT, 40):

                pygame.draw.line(
                    self.screen,
                    (255, 255, 255),
                    (x + 30, y),
                    (x + 30, y + 20),
                    3
                )

        self.player.draw(self.screen)

        for wave in self.waves:

            for enemy in wave:

                enemy.draw(self.screen)

        score = self.font.render(
            f"Score : {self.score}",
            True,
            (255, 255, 255)
        )

        self.screen.blit(score, (10, 10))

        pygame.display.flip()

        self.clock.tick(self.FPS)

    # ===================================================

    def close(self):
        if self.render_mode:
            pygame.quit()