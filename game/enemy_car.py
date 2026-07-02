import random

from .car import Car


class EnemyCar(Car):
    def __init__(self, lane_x, y, speed, color=(255, 0, 0)):
        lane = random.randint(0, len(lane_x) - 1)

        super().__init__(
            lane=lane,
            lane_x=lane_x,
            y=y,
            color=color
        )

        self.speed = speed

    def move(self):
        """Di chuyển xuống dưới."""
        self.y += self.speed

    def respawn(self, y, speed=None):
        """Đưa xe trở lại phía trên màn hình."""

        self.lane = random.randint(0, len(self.lane_x) - 1)
        self.y = y

        if speed is not None:
            self.speed = speed

        # Cập nhật ngay vị trí và rect
        self.update()

    def update(self):
        self.move()
        super().update()