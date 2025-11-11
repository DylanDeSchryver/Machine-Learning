# Tetris AI - Genetic Training System

## 🎯 Current Performance
**Baseline**: 10,136 rows cleared

## 🚀 Quick Commands

### Play with Best Model
```bash
python main.py TetriMind
```

### Train (Simple)
```bash
# Quick test (5-10 minutes)
python train.py

# Train 10 generations (2-3 hours)
python train.py 10

# Train 20 generations (4-6 hours) - Recommended overnight
python train.py 20
```

### Train (Advanced)
```bash
# Custom: 50 generations, 30 agents, 10 games each
python train.py 50 30 10
```

## 📊 How It Works

1. **Creates Population**: 20 AI agents with different weights
2. **Evaluates Fitness**: Each plays 5 games, scored by rows cleared
3. **Evolves**: Best agents survive, create mutated offspring
4. **Repeats**: Each generation improves on the last
5. **Auto-Saves**: Best model saved after each generation

## 💾 Saved Files

- `best_model.json` - Your best performing model (auto-loaded when playing)
- `training_history.json` - Progress tracking across all sessions
- Training continues from where you left off - never loses progress!

## 📈 Expected Improvements

| Generations | Expected Best Performance |
|-------------|--------------------------|
| 0 (baseline)| 10,136 rows             |
| 10          | 10,000-12,000 rows      |
| 20          | 12,000-14,000 rows      |
| 50          | 14,000-18,000 rows      |
| 100+        | 18,000-25,000+ rows     |

## 🔄 Multi-Day Training Workflow

**Week 1**:
- Day 1: `python train.py 10` (train overnight)
- Day 2: Test with `python main.py TetriMind`, then `python train.py 10`
- Day 3: Continue with `python train.py 10`

**Week 2**:
- Keep training 10 generations per day
- System automatically continues from best model
- Track progress in `training_history.json`

**Week 3+**:
- Train until performance plateaus
- You'll likely achieve 15,000-20,000+ rows!

## 🎓 Understanding the Genetics

### Features Being Optimized (10 weights):
1. **Aggregate Height** - Keep stack low
2. **Lines Cleared** - Prioritize completing lines
3. **Holes** - Avoid creating holes
4. **Bumpiness** - Maintain smooth surface
5. **Max Height** - Prevent tall stacks
6. **Wells** - Manage deep gaps
7. **Column Transitions** - Surface roughness
8. **Row Transitions** - Horizontal roughness
9. **Pit Depth** - Avoid deep pits
10. **Blocks Above Holes** - Prevent trapped blocks

### Evolution Strategy:
- **20% Elites**: Best agents preserved
- **40% Mutations**: Variations of best agents
- **40% Exploration**: Random new agents

## 📖 Full Documentation

See `TRAINING_GUIDE.md` for complete details, troubleshooting, and advanced options.

## ✅ Key Features

- ✅ Automatic checkpoint saving
- ✅ Multi-session training support
- ✅ Progress tracking and history
- ✅ No manual intervention needed
- ✅ Always preserves best model
- ✅ Can train over days/weeks

## 🎮 Ready to Start!

1. Test current model: `python main.py TetriMind`
2. Start training: `python train.py 10`
3. Come back in a few hours
4. Test improved model: `python main.py TetriMind`
5. Repeat!

Your AI will keep getting better with each training session! 🚀
