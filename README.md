# Reinforcement Learning with Verifiable Rewards (RLVR)

An implementation of Reinforcement Learning with Verifiable Rewards for training language models with objective, verifiable feedback.

## Overview

This project implements RLVR, a technique that uses automated verifiers to provide objective rewards for language model training, improving reasoning, planning, and instruction-following capabilities. Unlike preference-based RLHF, RLVR uses deterministic verification to ensure model outputs are demonstrably correct.

## Project Structure

```
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Dataset creation and preprocessing
│   ├── 02_verifier_development.ipynb  # Building verification systems
│   ├── 03_rlvr_training.ipynb         # Main RLVR training pipeline
│   └── 04_evaluation.ipynb            # Model evaluation and analysis
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── training_config.py         # Training configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # Dataset handling
│   │   └── preprocessing.py           # Data preprocessing utilities
│   ├── verifiers/
│   │   ├── __init__.py
│   │   ├── base_verifier.py           # Abstract verifier class
│   │   ├── code_verifier.py           # Code execution verification
│   │   ├── math_verifier.py           # Mathematical verification
│   │   └── logic_verifier.py          # Logical reasoning verification
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── base_reward.py             # Abstract reward class
│   │   ├── hybrid_reward.py           # Combined reward functions
│   │   └── reward_factory.py          # Reward function factory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── language_model.py          # Language model wrapper
│   │   └── policy.py                  # RL policy implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── rlvr_trainer.py            # Main RLVR trainer
│   │   ├── ppo_trainer.py             # PPO implementation
│   │   └── experience_buffer.py       # Experience replay buffer
│   └── utils/
│       ├── __init__.py
│       ├── logging.py                 # Logging utilities
│       └── metrics.py                 # Training metrics
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
└── README.md                         # This file
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ReinforcementLearningHybridReward

# Create virtual environment
python -m venv rlvr_env
source rlvr_env/bin/activate  # On Windows: rlvr_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

1. **Data Preparation**: Run `notebooks/01_data_preparation.ipynb` to create your training dataset
2. **Verifier Development**: Use `notebooks/02_verifier_development.ipynb` to build verification systems
3. **Training**: Execute `notebooks/03_rlvr_training.ipynb` to start RLVR training
4. **Evaluation**: Analyze results with `notebooks/04_evaluation.ipynb`

## Core Concepts

### RLVR Process

1. **Instruction Generation**: Create diverse, challenging instructions
2. **Model Response**: Generate responses using the current policy
3. **Verification**: Apply automated verifiers to check correctness
4. **Reward Calculation**: Compute objective rewards based on verification results
5. **Policy Update**: Update the model using reinforcement learning

### Verification Types

- **Code Verification**: Execute generated code and verify outputs
- **Mathematical Verification**: Check mathematical correctness and reasoning
- **Logical Verification**: Validate logical consistency and reasoning steps
- **Custom Verifiers**: Extend with domain-specific verification logic

### Reward Architecture

The reward system is designed to be modular and easily extensible:

- **Base Reward**: Abstract class for all reward functions
- **Hybrid Rewards**: Combine multiple verification signals
- **Reward Factory**: Easy creation and management of reward functions

## Configuration

Training parameters can be configured in `src/config/training_config.py`:

```python
# Example configuration
TRAINING_CONFIG = {
    "model_name": "gpt2-medium",
    "learning_rate": 1e-5,
    "batch_size": 4,
    "max_length": 512,
    "ppo_epochs": 4,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

- [VerIF: Verification Engineering for RL in Instruction Following](https://arxiv.org/abs/2506.09942)
- [IFDecorator: Wrapping Instruction Following RL with Verifiable Rewards](https://arxiv.org/abs/2508.04632)
- [Labelbox RLVR Solutions](https://labelbox.com/solutions/reinforcement-learning-with-verifiable-rewards/)
