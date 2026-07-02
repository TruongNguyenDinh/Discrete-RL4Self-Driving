# car.py

import pygame


class Car:
    def __init__(self, lane, lane_x, y, color):

        self.lane = lane
        self.lane_x = lane_x

        self.x = lane_x[lane]
        self.y = y

        self.width = 60
        self.height = 100

        self.color = color

        self.rect = pygame.Rect(
            self.x,
            self.y,
            self.width,
            self.height
        )

    def move_left(self):
        if self.lane > 0:
            self.lane -= 1

    def move_right(self):
        if self.lane < len(self.lane_x) - 1:
            self.lane += 1

    def stay(self):
        pass

    def step(self, action):
        """
        0 = Left
        1 = Stay
        2 = Right
        """

        if action == 0:
            self.move_left()

        elif action == 2:
            self.move_right()

    def update(self):
        """Cập nhật vị trí sau khi đổi làn"""

        self.x = self.lane_x[self.lane]

        self.rect.x = self.x
        self.rect.y = self.y

    def draw(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            self.rect
        )

    def collide(self, other):
        return self.rect.colliderect(other.rect)