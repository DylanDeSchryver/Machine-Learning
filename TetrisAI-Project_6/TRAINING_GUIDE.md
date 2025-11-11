# Tetris AI Training Guide

## Overview

Your custom AI model now includes a **Genetic Algorithm training system** that allows you to evolve better-performing models over multiple generations and training sessions.

**Current Performance**: 10,136 rows cleared (baseline)

---

## Quick Start

### 1. Play with Current Best Model
```bash
python main.py TetriMind
```
This automatically loads the best saved model (or uses default weights if no saved model exists).

### 2. Train for One Generation (Quick Test)
```bash
python -c "from src.custom_model import train_one_generation; train_one_generation()"
```
- Trains 20 agents
- Each plays 5 games
- Takes ~5-10 minutes
- Saves best model automatically

### 3. Train for Multiple Generations (Overnight Training)
```bash
python -c "from src.custom_model import train_multiple_generations; train_multiple_generations(generations=20)"
```
- Trains 20 generations
- Each generation has 20 agents playing 5 games
- Total: 2,000 games
- Takes several hours
- Perfect for overnight training

---

## How It Works

### Genetic Algorithm Process

1. **Generation 1**: Creates 20 agents with random weights (plus your baseline model)
2. **Evaluation**: Each agent plays 5 games, fitness = avg rows cleared
3. **Selection**: Top 4 agents are kept as "elites"
4. **Mutation**: Create variations of the best agents
5. **Next Generation**: Repeat with improved population

### Evolution Strategy

- **Elites (20%)**: Best agents preserved unchanged
- **Mutations (40%)**: Moderate variations of best agents
- **Exploration (40%)**: Random weights for diversity

### Fitness Function
```
fitness = avg_rows_cleared + (avg_pieces_dropped * 0.1)
```
Prioritizes clearing rows while rewarding survival time.

---

## Training Over Multiple Days

### Day 1: Initial Training
```bash
# Train for 10 generations (2-3 hours)
python -c "from src.custom_model import train_multiple_generations; train_multiple_generations(generations=10)"
```

### Day 2: Continue Training
```bash
# Train 10 more generations - automatically loads best model from Day 1
python -c "from src.custom_model import train_multiple_generations; train_multiple_generations(generations=10)"
```

### Day 3+: Keep Improving
```bash
# Keep training - it always continues from the best saved model
python -c "from src.custom_model import train_multiple_generations; train_multiple_generations(generations=10)"
```

**The system automatically**:
- ✅ Loads the best model from previous sessions
- ✅ Continues generation numbering
- ✅ Saves improvements
- ✅ Tracks training history

---

## Saved Files

### `src/best_model.json`
Contains the best performing model:
```json
{
  "weights": {
    "aggregate_height": -0.510066,
    "lines_cleared": 0.760666,
    ...
  },
  "fitness": 10245.5,
  "generation": 15,
  "rows_cleared": 10136,
  "timestamp": "2025-11-10T18:45:00"
}
```

### `src/training_history.json`
Tracks progress over all generations:
```json
{
  "generations": [1, 2, 3, ...],
  "best_fitness_per_gen": [8500, 9200, 9800, ...],
  "avg_fitness_per_gen": [6000, 7000, 7500, ...]
}
```

---

## Customizing Training

### Quick Training (Testing)
```python
from src.custom_model import train_one_generation

train_one_generation(
    population_size=10,  # Fewer agents
    num_games=3,         # Fewer games per agent
    elite_count=2        # Fewer elites
)
# Takes ~2-3 minutes
```

### Intensive Training (Best Results)
```python
from src.custom_model import train_multiple_generations

train_multiple_generations(
    generations=50,      # Many generations
    population_size=30,  # More agents
    num_games=10         # More games per agent
)
# Takes 10-20 hours, but gives best results
```

### Balanced Training (Recommended)
```python
from src.custom_model import train_multiple_generations

train_multiple_generations(
    generations=20,      # Good number of generations
    population_size=20,  # Standard population
    num_games=5          # Enough for reliable evaluation
)
# Takes 3-5 hours
```

---

## Expected Results

### Generation 1 (Random)
- Best: 5,000-8,000 rows
- Avg: 2,000-4,000 rows
- Your baseline (10,136) will likely be the best

### Generation 5-10
- Best: 10,000-12,000 rows
- Avg: 6,000-8,000 rows
- Starting to see improvements

### Generation 20-30
- Best: 12,000-15,000 rows
- Avg: 8,000-10,000 rows
- Significant improvements

### Generation 50+
- Best: 15,000-20,000+ rows
- Avg: 10,000-12,000 rows
- Near-optimal performance

---

## Tips for Best Results

### 1. **Start with Overnight Training**
Run 20-30 generations overnight to get a good baseline improvement.

### 2. **Train in Batches**
Train 10 generations at a time over several days rather than 100 at once.

### 3. **Monitor Progress**
Check `training_history.json` to see if fitness is still improving:
- If fitness plateaus for 10+ generations, you may have reached optimal weights
- If still improving, keep training!

### 4. **Test Your Model**
After each training session, test it:
```bash
python main.py TetriMind
```

### 5. **Backup Your Best Model**
```bash
cp src/best_model.json src/best_model_backup_gen20.json
```

---

## Troubleshooting

### "No improvement after many generations"
- This is normal! You may have found near-optimal weights
- Try increasing mutation rate or population diversity
- Your baseline (10,136) is already very good

### "Training is too slow"
- Reduce `population_size` to 10-15
- Reduce `num_games` to 3
- Train fewer generations at a time

### "Model performs worse after training"
- Check `best_model.json` - it only saves if better than previous
- Your baseline is already research-proven, so improvements may be marginal
- The genetic algorithm explores the space around your good baseline

### "Want to start fresh"
```bash
rm src/best_model.json
rm src/training_history.json
```

---

## Advanced: Analyzing Results

### View Training Progress
```python
import json

with open('src/training_history.json', 'r') as f:
    history = json.load(f)

print(f"Generations trained: {len(history['generations'])}")
print(f"Best fitness: {max(history['best_fitness_per_gen'])}")
print(f"Improvement: {history['best_fitness_per_gen'][-1] - history['best_fitness_per_gen'][0]}")
```

### Compare Weights
```python
import json

with open('src/best_model.json', 'r') as f:
    best = json.load(f)

print("Best weights:")
for key, value in best['weights'].items():
    print(f"  {key}: {value:.4f}")
```

---

## Summary

Your AI model is now equipped with evolutionary training! The system:

✅ **Automatically saves** the best model after each generation
✅ **Continues training** from where you left off
✅ **Tracks progress** across all training sessions
✅ **Works over multiple days** - train a bit each day
✅ **Never loses progress** - best model is always preserved

**Recommended workflow**:
1. Train 10 generations (2-3 hours)
2. Test the model
3. Train 10 more generations the next day
4. Repeat until performance plateaus
5. You'll likely achieve 12,000-15,000+ rows cleared!

Good luck with your training! 🎮🧬
