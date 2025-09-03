# RLVR Implementation Guide

## Overview

This guide explains how to use the Reinforcement Learning with Verifiable Rewards (RLVR) codebase. The implementation is designed to be modular, extensible, and production-ready.

## Quick Start

### 1. Installation

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

### 2. Run the Notebooks

1. **Data Preparation** (`notebooks/01_data_preparation.ipynb`)

   - Generate synthetic training data
   - Validate and preprocess data
   - Split into train/validation/test sets

2. **Verifier Development** (`notebooks/02_verifier_development.ipynb`)

   - Test different verifiers
   - Validate verification logic
   - Batch verification testing

3. **RLVR Training** (`notebooks/03_rlvr_training.ipynb`)

   - Initialize components
   - Start training process
   - Monitor progress

4. **Evaluation** (`notebooks/04_evaluation.ipynb`)
   - Analyze training results
   - Generate visualizations
   - Performance metrics

## Architecture Overview

### Core Components

1. **Configuration** (`src/config/`)

   - `TrainingConfig`: Centralized configuration management
   - Pydantic validation for type safety
   - Easy configuration loading/saving

2. **Verifiers** (`src/verifiers/`)

   - `BaseVerifier`: Abstract interface for all verifiers
   - `CodeVerifier`: Executes and verifies code
   - `MathVerifier`: Checks mathematical correctness
   - `LogicVerifier`: Validates logical reasoning

3. **Rewards** (`src/rewards/`)

   - `BaseReward`: Abstract reward function interface
   - `HybridReward`: Combines multiple verification signals
   - `RewardFactory`: Easy reward function creation

4. **Models** (`src/models/`)

   - `LanguageModel`: Unified interface for language models
   - `Policy`: Policy wrapper for RL training

5. **Training** (`src/training/`)

   - `RLVRTrainer`: Main training orchestrator
   - `PPOTrainer`: PPO algorithm implementation
   - `ExperienceBuffer`: Experience replay buffer

6. **Utilities** (`src/utils/`)
   - `MetricsTracker`: Training metrics and visualization
   - `Logging`: Structured logging utilities

### Data Flow

```
Instruction → Model Generation → Verification → Reward Computation → Policy Update
     ↓              ↓                ↓              ↓                ↓
Training Data → Language Model → Verifiers → Reward Function → PPO Trainer
```

## Key Features

### 1. Modular Architecture

- **Easy to extend**: Add new verifiers by inheriting from `BaseVerifier`
- **Configurable rewards**: Modify reward functions without changing core logic
- **Pluggable models**: Support for different language model architectures

### 2. Production Ready

- **Comprehensive logging**: Structured logging with multiple backends
- **Error handling**: Robust error handling throughout the pipeline
- **Configuration validation**: Type-safe configuration with Pydantic
- **Metrics tracking**: Detailed training metrics and visualization

### 3. Verification Types

- **Code Verification**: Execute generated code and verify outputs
- **Mathematical Verification**: Check mathematical correctness and reasoning
- **Logical Verification**: Validate logical consistency and arguments
- **Extensible**: Easy to add new verification types

### 4. Reward Architecture

- **Hybrid Rewards**: Combine multiple verification signals
- **Configurable Weights**: Adjust importance of different components
- **Reward Shaping**: Apply reward shaping for better learning
- **Confidence Scoring**: Uncertainty-aware reward computation

## Usage Examples

### Basic Training

```python
from src.config.training_config import get_fast_config
from src.models.language_model import LanguageModel
from src.verifiers.code_verifier import CodeVerifier
from src.rewards.reward_factory import RewardFactory
from src.training.rlvr_trainer import RLVRTrainer

# Initialize components
config = get_fast_config()
language_model = LanguageModel(config.model)
verifiers = [CodeVerifier()]
reward_function = RewardFactory().create_reward_from_preset("default_hybrid")

# Create trainer
trainer = RLVRTrainer(config, language_model, verifiers, reward_function)

# Start training
trainer.train(training_data)
```

### Custom Verifier

```python
from src.verifiers.base_verifier import BaseVerifier, VerificationOutput, VerificationResult

class CustomVerifier(BaseVerifier):
    def verify(self, instruction, model_output, expected_output=None, context=None):
        # Custom verification logic
        if "custom_check" in model_output:
            return VerificationOutput(
                result=VerificationResult.CORRECT,
                score=1.0,
                details={"custom_check": True}
            )
        else:
            return VerificationOutput(
                result=VerificationResult.INCORRECT,
                score=0.0,
                details={"custom_check": False}
            )
```

### Custom Reward Function

```python
from src.rewards.base_reward import BaseReward, RewardOutput, RewardType

class CustomReward(BaseReward):
    def compute_reward(self, instruction, model_output, verification_outputs, context=None):
        # Custom reward computation
        base_reward = sum(v.score for v in verification_outputs) / len(verification_outputs)

        # Add custom bonus
        if "bonus_keyword" in model_output:
            base_reward += 0.2

        return RewardOutput(
            reward=base_reward,
            reward_type=RewardType.CUSTOM,
            components={"base": base_reward},
            metadata={"custom_bonus": True},
            confidence=0.8
        )
```

## Configuration

### Training Configuration

```python
from src.config.training_config import TrainingConfig

config = TrainingConfig(
    model=ModelConfig(
        model_name="gpt2-medium",
        max_length=512,
        device="auto"
    ),
    learning_rate=1e-5,
    batch_size=4,
    num_episodes=1000,
    ppo_epochs=4,
    clip_epsilon=0.2
)
```

### Verifier Configuration

```python
code_verifier = CodeVerifier(config={
    "timeout": 30,
    "safe_mode": True,
    "allowed_modules": ["math", "random", "datetime"]
})
```

### Reward Configuration

```python
reward_function = HybridReward(config={
    "verification_weight": 0.7,
    "quality_weight": 0.2,
    "diversity_weight": 0.05,
    "efficiency_weight": 0.05
})
```

## Best Practices

### 1. Data Preparation

- **Validate data quality**: Use the provided validation tools
- **Balance datasets**: Ensure equal representation of different task types
- **Preprocess consistently**: Apply the same preprocessing to train/val/test sets

### 2. Verifier Development

- **Test thoroughly**: Verify your verifiers work correctly
- **Handle edge cases**: Consider error conditions and timeouts
- **Document behavior**: Clearly document what your verifier checks

### 3. Reward Design

- **Start simple**: Begin with basic verification-based rewards
- **Iterate gradually**: Add complexity incrementally
- **Monitor behavior**: Watch for reward hacking or exploitation

### 4. Training

- **Start small**: Use fast configurations for initial testing
- **Monitor metrics**: Track verification accuracy and reward trends
- **Save checkpoints**: Regular checkpointing for recovery

## Troubleshooting

### Common Issues

1. **Memory Issues**

   - Reduce batch size
   - Use smaller models
   - Enable gradient checkpointing

2. **Slow Training**

   - Use faster configurations
   - Reduce verification complexity
   - Optimize data loading

3. **Poor Performance**
   - Check data quality
   - Verify reward function
   - Adjust hyperparameters

### Debugging

- Enable debug logging: `setup_logging(log_level="DEBUG")`
- Check verification outputs: `verifier.get_verification_stats()`
- Monitor reward components: `reward_function.get_reward_stats()`

## Extending the Codebase

### Adding New Verifiers

1. Inherit from `BaseVerifier`
2. Implement the `verify` method
3. Add to verifier registry
4. Update tests

### Adding New Reward Functions

1. Inherit from `BaseReward`
2. Implement `compute_reward` method
3. Register with `RewardFactory`
4. Add configuration presets

### Adding New Models

1. Extend `LanguageModel` class
2. Implement required methods
3. Add model-specific configuration
4. Update model loading logic

## Performance Optimization

### Training Speed

- Use smaller models for initial experiments
- Reduce verification complexity
- Optimize data loading with prefetching
- Use mixed precision training

### Memory Efficiency

- Gradient checkpointing
- Dynamic batching
- Memory-efficient verifiers
- Regular garbage collection

### Scalability

- Distributed training support
- Multi-GPU training
- Efficient data pipelines
- Caching mechanisms

## Contributing

1. Follow the existing code style
2. Add comprehensive tests
3. Update documentation
4. Use type hints
5. Add logging for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [VerIF: Verification Engineering for RL in Instruction Following](https://arxiv.org/abs/2506.09942)
- [IFDecorator: Wrapping Instruction Following RL with Verifiable Rewards](https://arxiv.org/abs/2508.04632)
- [Labelbox RLVR Solutions](https://labelbox.com/solutions/reinforcement-learning-with-verifiable-rewards/)
