import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame_initialized = False
font = None

def init_pygame():
    global pygame_initialized, font
    if not pygame_initialized:
        pygame.init()
        font = pygame.font.SysFont('arial', 25)
        pygame_initialized = True

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 1000

class SnakeGameAI:

    def __init__(self, w=640, h=480, headless=False):
        self.w = w
        self.h = h
        self.headless = headless
        if not headless:
            init_pygame()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
        self.total_games = 0
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.foods = []
        self.total_games += 1
        self.apple_count = max(1, 6 - (self.total_games // 200))
        for _ in range(self.apple_count):
            self._place_food()
            
        self.frame_iteration = 0
        # for loop detection
        self.recent_positions = []

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        new_food = Point(x, y)
        if new_food not in self.snake and new_food not in self.foods:
            self.foods.append(new_food)
        else:
            self._place_food()

    def play_step(self, action):
        # main loop
        self.frame_iteration += 1
        if not self.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -15
            return reward, game_over, self.score

        reward = 0
        loop_penalty = self._calculate_loop_penalty()
        reward += loop_penalty

        ate_food = False
        for food in self.foods[:]:
            if self.head == food:
                self.score += 1
                reward = 10
                self.foods.remove(food)
                self._place_food()
                ate_food = True
                break
        if not ate_food:
            self.snake.pop()

        if not self.headless:
            self._update_ui()
            self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        for food in self.foods:
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _calculate_loop_penalty(self):
        penalty = 0
        self.recent_positions.append(self.head)
        if len(self.recent_positions) > 15:
            self.recent_positions.pop(0)
        if len(self.recent_positions) >= 10:
            head_count = self.recent_positions.count(self.head)
            if head_count > 1:
                penalty = -5 * (head_count - 1)
        if len(self.recent_positions) >= 8:
            unique_positions = len(set(self.recent_positions[-8:]))
            if unique_positions <= 3:
                penalty -= 8
        return penalty