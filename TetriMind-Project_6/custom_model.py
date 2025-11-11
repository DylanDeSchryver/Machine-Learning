from copy import copy, deepcopy
import numpy as np
import pygame
from piece import BODIES, Piece
from board import Board
from random import randint
import json
import os
from datetime import datetime
import random

"""
Enhanced AI Model for Tetris with Genetic Algorithm Training

------
1. To play with the current best model:
   python main.py TetriMind

2. To train for one generation (10 games per agent, 20 agents):
   python -c "from custom_model import train_one_generation; train_one_generation()"

3. To train for multiple generations:
   python -c "from custom_model import train_multiple_generations; train_multiple_generations(generations=10)"

4. To continue training from saved checkpoint:
   Just run training again - it automatically loads the best model and continues

The best model is automatically saved to: best_model.json
Training history is saved to: training_history.json
"""

class CUSTOM_AI_MODEL:
    def __init__(self, weights=None):
        if weights is not None:
            self.weights = weights
        else:
            best_weights = load_best_model()
            if best_weights:
                self.weights = best_weights
                print("Loaded best model from file")
            else:
                self.weights = {
                    'aggregate_height': -0.510066,
                    'lines_cleared': 0.760666,
                    'holes': -0.35663,
                    'bumpiness': -0.184483,
                    'max_height': -0.5,
                    'wells': -0.3,
                    'column_transitions': -0.1,
                    'row_transitions': -0.1,
                    'pit_depth': -0.2,
                    'blocks_above_holes': -0.4
                }
        
        self.fitness_scores = []
        self.avg_fitness = 0
    
    def get_best_move(self, board, piece, depth=1):

        best_x = -1
        best_piece = None
        best_score = float('-inf')
        
        current_piece = piece
        for rotation in range(4):
            current_piece = current_piece.get_next_rotation()
            
            for x in range(board.width):
                try:
                    y = board.drop_height(current_piece, x)
                except:
                    continue
                
                score = self.evaluate_move(board, current_piece, x, y)
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_piece = current_piece
        
        return best_x, best_piece
    
    def evaluate_move(self, board, piece, x, y):
       
        board_copy = deepcopy(board.board)
        
        for pos in piece.body:
            try:
                board_copy[y + pos[1]][x + pos[0]] = True
            except:
                return float('-inf')
        
        features = self.extract_features(board_copy)
        
        score = sum(self.weights[key] * features[key] for key in self.weights.keys())
        
        return score
    
    def extract_features(self, board):
   
        features = {}
        
        heights = self.get_column_heights(board)
        
        features['aggregate_height'] = sum(heights)
        
        features['max_height'] = max(heights) if heights else 0
        
        features['lines_cleared'] = self.count_complete_lines(board)
        
        features['holes'] = self.count_holes(board, heights)
        
        features['bumpiness'] = self.calculate_bumpiness(heights)
        
        features['wells'] = self.calculate_wells(heights)
        
        features['column_transitions'] = self.count_column_transitions(board, heights)
        
        features['row_transitions'] = self.count_row_transitions(board)
        
        features['pit_depth'] = self.calculate_pit_depth(heights)
        
        features['blocks_above_holes'] = self.count_blocks_above_holes(board, heights)
        
        return features
    
    def get_column_heights(self, board):
       
        heights = []
        for col in range(len(board[0])):
            height = 0
            for row in range(len(board)):
                if board[row][col]:
                    height = row + 1
            heights.append(height)
        return heights
    
    def count_complete_lines(self, board):
       
        complete_lines = 0
        for row in board:
            if all(row):
                complete_lines += 1
        return complete_lines
    
    def count_holes(self, board, heights):
       
        holes = 0
        for col in range(len(board[0])):
            found_block = False
            for row in range(len(board) - 1, -1, -1):
                if board[row][col]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes
    
    def calculate_bumpiness(self, heights):
       
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness
    
    def calculate_wells(self, heights):
        wells = 0
        for i in range(len(heights)):
            if i == 0:
                if len(heights) > 1:
                    wells += max(0, heights[1] - heights[0])
            elif i == len(heights) - 1:
                wells += max(0, heights[-2] - heights[-1])
            else:
                left_diff = heights[i - 1] - heights[i]
                right_diff = heights[i + 1] - heights[i]
                if left_diff > 0 and right_diff > 0:
                    wells += min(left_diff, right_diff)
        return wells
    
    def count_column_transitions(self, board, heights):
        transitions = 0
        for col in range(len(board[0])):
            for row in range(len(board) - 1):
                if board[row][col] != board[row + 1][col]:
                    transitions += 1
        return transitions
    
    def count_row_transitions(self, board):
        transitions = 0
        for row in range(len(board)):
            for col in range(len(board[0]) - 1):
                if board[row][col] != board[row][col + 1]:
                    transitions += 1
        return transitions
    
    def calculate_pit_depth(self, heights):
        if not heights:
            return 0
        
        max_pit = 0
        for i in range(len(heights)):
            if i == 0:
                if len(heights) > 1:
                    pit = heights[1] - heights[0]
                    max_pit = max(max_pit, pit)
            elif i == len(heights) - 1:
                pit = heights[-2] - heights[-1]
                max_pit = max(max_pit, pit)
            else:
                left_diff = heights[i - 1] - heights[i]
                right_diff = heights[i + 1] - heights[i]
                if left_diff > 0 and right_diff > 0:
                    pit = min(left_diff, right_diff)
                    max_pit = max(max_pit, pit)
        return max_pit
    
    def count_blocks_above_holes(self, board, heights):
        blocks_above = 0
        for col in range(len(board[0])):
            hole_found = False
            for row in range(len(board) - 1, -1, -1):
                if not board[row][col] and row < heights[col] - 1:
                    hole_found = True
                elif hole_found and board[row][col]:
                    blocks_above += 1
        return blocks_above



def load_best_model():
    filepath = 'best_model.json'
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data['weights']
        except:
            return None
    return None


def save_best_model(weights, fitness, generation, rows_cleared):
    filepath = 'best_model.json'
    data = {
        'weights': weights,
        'fitness': fitness,
        'generation': generation,
        'rows_cleared': rows_cleared,
        'timestamp': datetime.now().isoformat()
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved best model (fitness: {fitness:.2f}, rows: {rows_cleared})")


def load_training_history():
    filepath = 'training_history.json'
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            return {'generations': [], 'best_fitness_per_gen': [], 'avg_fitness_per_gen': []}
    return {'generations': [], 'best_fitness_per_gen': [], 'avg_fitness_per_gen': []}


def save_training_history(history):
    filepath = 'training_history.json'
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)


def create_random_weights():
    return {
        'aggregate_height': random.uniform(-1.0, 0.0),
        'lines_cleared': random.uniform(0.0, 1.5),
        'holes': random.uniform(-1.0, 0.0),
        'bumpiness': random.uniform(-0.5, 0.0),
        'max_height': random.uniform(-1.0, 0.0),
        'wells': random.uniform(-0.5, 0.0),
        'column_transitions': random.uniform(-0.3, 0.0),
        'row_transitions': random.uniform(-0.3, 0.0),
        'pit_depth': random.uniform(-0.5, 0.0),
        'blocks_above_holes': random.uniform(-1.0, 0.0)
    }


def mutate_weights(weights, mutation_rate=0.15, mutation_scale=0.2):
    mutated = weights.copy()
    for key in mutated:
        if random.random() < mutation_rate:
            # Add Gaussian noise
            noise = random.gauss(0, mutation_scale)
            mutated[key] = mutated[key] * (1 + noise)
            
            # Keep weights in reasonable ranges
            if key == 'lines_cleared':
                mutated[key] = max(0.0, min(2.0, mutated[key]))
            else:
                mutated[key] = max(-2.0, min(0.5, mutated[key]))
    
    return mutated


def crossover_weights(parent1_weights, parent2_weights):

    child_weights = {}
    for key in parent1_weights:
        if random.random() < 0.5:
            child_weights[key] = parent1_weights[key]
        else:
            child_weights[key] = parent2_weights[key]
    return child_weights


def evaluate_agent(agent, num_games=5, verbose=False):

    from game import Game
    
    total_rows = 0
    total_pieces = 0
    best_rows = 0
    
    for game_num in range(num_games):
        game = Game("TetriMind", agent=agent)
        pieces_dropped, rows_cleared = game.run_no_visual()
        
        total_rows += rows_cleared
        total_pieces += pieces_dropped
        best_rows = max(best_rows, rows_cleared)
        
        if verbose:
            print(f"  Game {game_num + 1}/{num_games}: {rows_cleared} rows, {pieces_dropped} pieces")
    
    avg_rows = total_rows / num_games
    avg_pieces = total_pieces / num_games
    
    fitness = avg_rows + (avg_pieces * 0.1)
    
    return fitness, best_rows, avg_rows


def train_one_generation(population_size=20, num_games=5, elite_count=4):

    print("\n" + "="*70)
    print("STARTING NEW GENERATION")
    print("="*70)
    
    history = load_training_history()
    current_gen = len(history['generations']) + 1
    
    population = []
    
    best_weights = load_best_model()
    
    if best_weights and current_gen > 1:
        print(f"Generation {current_gen}: Continuing from saved best model")
        population.append(CUSTOM_AI_MODEL(best_weights))
        
        for i in range(elite_count - 1):
            mutated = mutate_weights(best_weights, mutation_rate=0.1, mutation_scale=0.15)
            population.append(CUSTOM_AI_MODEL(mutated))
        
        for i in range(population_size - elite_count):
            if i < (population_size - elite_count) // 2:
                mutated = mutate_weights(best_weights, mutation_rate=0.2, mutation_scale=0.25)
            else:
                mutated = create_random_weights()
            population.append(CUSTOM_AI_MODEL(mutated))
    else:
        print(f"Generation {current_gen}: Creating initial random population")
        for i in range(population_size):
            if i == 0 and best_weights:
                population.append(CUSTOM_AI_MODEL(best_weights))
            else:
                population.append(CUSTOM_AI_MODEL(create_random_weights()))
    
    print(f"\nEvaluating {population_size} agents ({num_games} games each)...")
    fitness_scores = []
    
    for idx, agent in enumerate(population):
        print(f"\nAgent {idx + 1}/{population_size}:")
        fitness, best_rows, avg_rows = evaluate_agent(agent, num_games=num_games, verbose=True)
        fitness_scores.append((fitness, best_rows, avg_rows, agent))
        print(f"  → Fitness: {fitness:.2f}, Best: {best_rows} rows, Avg: {avg_rows:.1f} rows")
    
    fitness_scores.sort(reverse=True, key=lambda x: x[0])
    
    best_fitness, best_rows, avg_rows, best_agent = fitness_scores[0]
    avg_fitness = sum(f[0] for f in fitness_scores) / len(fitness_scores)
    
    print("\n" + "="*70)
    print(f"GENERATION {current_gen} COMPLETE")
    print("="*70)
    print(f"Best Agent  - Fitness: {best_fitness:.2f}, Best Game: {best_rows} rows")
    print(f"Avg Fitness - {avg_fitness:.2f}")
    print(f"Top 5 Agents: {[f'{f[0]:.1f}' for f in fitness_scores[:5]]}")
    
    save_best_model(best_agent.weights, best_fitness, current_gen, best_rows)
    
    history['generations'].append(current_gen)
    history['best_fitness_per_gen'].append(best_fitness)
    history['avg_fitness_per_gen'].append(avg_fitness)
    save_training_history(history)
    
    return best_agent, best_fitness


def train_multiple_generations(generations=10, population_size=20, num_games=5):

    print("\n" + "="*70)
    print(f"TRAINING FOR {generations} GENERATIONS")
    print("="*70)
    print(f"Population size: {population_size}")
    print(f"Games per agent: {num_games}")
    print(f"Total games: {generations * population_size * num_games}")
    print("="*70)
    
    start_time = datetime.now()
    
    for gen in range(generations):
        best_agent, best_fitness = train_one_generation(
            population_size=population_size,
            num_games=num_games
        )
        
        elapsed = datetime.now() - start_time
        print(f"\nElapsed time: {elapsed}")
        print(f"Estimated remaining: {elapsed / (gen + 1) * (generations - gen - 1)}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {datetime.now() - start_time}")
    print(f"\nBest model saved to: best_model.json")
    print(f"Training history saved to: training_history.json")
    print("\nTo use the best model, run: python main.py TetriMind")

if __name__ == "__main__":
    print("Training one generation...")
    train_one_generation(population_size=20, num_games=5)
