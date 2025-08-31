import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer

MAX_MEMORY = 1000
BATCH_SIZE = 32
LR = 0.001
# agent setup

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(12, 64, 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, device=self.device)
        print(f"Using device: {self.device}")

    def get_state(self, game):
        # get state
        head = game.snake[0]
        directions = [
            Point(0, -20),
            Point(20, 0),
            Point(0, 20),
            Point(-20, 0),
        ]
        state = []
        for direction in directions:
            current = Point(head.x + direction.x, head.y + direction.y)
            danger = game.is_collision(current) or current in game.snake[1:]
            state.append(1.0 if danger else 0.0)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        state.extend([dir_l, dir_r, dir_u, dir_d])

        if game.foods:
            closest_food = min(game.foods, key=lambda f: abs(f.x - head.x) + abs(f.y - head.y))
            food_left = closest_food.x < head.x
            food_right = closest_food.x > head.x
            food_up = closest_food.y < head.y
            food_down = closest_food.y > head.y
        else:
            food_left = food_right = food_up = food_down = False
        state.extend([food_left, food_right, food_up, food_down])
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if self.epsilon < 0:
            self.epsilon = 0
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move