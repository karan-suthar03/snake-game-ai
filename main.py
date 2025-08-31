import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from agent import Agent

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    
    game = SnakeGameAI()
    
    try:
        while True:
            game.reset()
            
            while True:
                state_old = agent.get_state(game)
                final_move = agent.get_action(state_old)
                reward, done, score = game.play_step(final_move)
                state_new = agent.get_state(game)
                
                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                agent.remember(state_old, final_move, reward, state_new, done)
                
                if done:
                    break
            
            agent.n_games += 1
            
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                print(f'🎉 NEW RECORD! Game {agent.n_games}: Score {score}')
            
            if agent.n_games % 50 == 0:
                avg_score = sum(plot_scores[-10:]) / len(plot_scores[-10:]) if plot_scores else 0
                print(f' Game {agent.n_games}, Score: {score}, Avg(10): {avg_score:.1f}, Record: {record}, Epsilon: {agent.epsilon:.1f}')
                print(f'   Memory: {len(agent.memory)}/1000')

    except KeyboardInterrupt:
        print("\n Training interrupted!")
        print(f" Final Stats - Games: {agent.n_games}, Best Score: {record}")
        agent.model.save()
        print("Model saved!")

if __name__ == '__main__':
    train()