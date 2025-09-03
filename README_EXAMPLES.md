# RLVR Example Scripts

This directory contains example scripts that demonstrate how to use the Reinforcement Learning with Verifiable Rewards (RLVR) system end-to-end.

## Quick Start

### Option 1: Run Everything at Once

```bash
python example_end_to_end.py
```

This will run the complete RLVR pipeline:

1. Generate synthetic training data
2. Test verifiers
3. Train the model with verifiable rewards
4. Evaluate results

### Option 2: Run Step by Step

```bash
# Step 1: Prepare data
python 01_data_preparation.py

# Step 2: Test verifiers
python 02_verifier_development.py

# Step 3: Train the model
python 03_rlvr_training.py

# Step 4: Evaluate results
python 04_evaluation.py
```

## Script Descriptions

### `example_end_to_end.py`

**Complete end-to-end example**

This script demonstrates the entire RLVR pipeline in one go:

- Generates synthetic training data (code generation, math reasoning, logic reasoning)
- Initializes and tests verifiers (code, math, logic)
- Sets up hybrid reward function
- Runs RLVR training with PPO
- Evaluates and saves results

**Usage:**

```bash
python example_end_to_end.py
```

**Output:**

- Training data in `data/processed/`
- Results in `results/`
- Comprehensive logs and metrics

### `01_data_preparation.py`

**Data generation and preparation**

This script:

- Generates synthetic training examples for different task types
- Validates data quality
- Preprocesses and filters data
- Splits into train/validation/test sets
- Creates training datasets

**Usage:**

```bash
python 01_data_preparation.py
```

**Output:**

- `data/processed/train_data.json`
- `data/processed/val_data.json`
- `data/processed/test_data.json`

### `02_verifier_development.py`

**Verifier testing and validation**

This script:

- Tests individual verifiers with known examples
- Runs batch verification on training data
- Analyzes verification performance
- Saves verification results

**Usage:**

```bash
python 02_verifier_development.py
```

**Output:**

- `results/verification_results.json`
- Performance metrics and analysis

### `03_rlvr_training.py`

**RLVR training pipeline**

This script:

- Loads prepared training data
- Initializes language model, verifiers, and reward function
- Demonstrates verification on sample examples
- Runs complete RLVR training
- Saves training results

**Usage:**

```bash
python 03_rlvr_training.py
```

**Output:**

- `results/training_results.json`
- `results/training_analysis.json`
- Training logs and metrics

### `04_evaluation.py`

**Results evaluation and visualization**

This script:

- Loads training and verification results
- Analyzes performance metrics
- Generates visualizations (plots)
- Creates comprehensive evaluation report
- Compares with baseline metrics

**Usage:**

```bash
python 04_evaluation.py
```

**Output:**

- `results/plots/` (visualizations)
- `results/evaluation_report.json`
- Performance analysis and recommendations

## Generated Files

After running the scripts, you'll find:

### Data Files

```
data/processed/
├── train_data.json      # Training examples
├── val_data.json        # Validation examples
└── test_data.json       # Test examples
```

### Results Files

```
results/
├── training_results.json     # Raw training results
├── training_analysis.json    # Training analysis
├── verification_results.json # Verifier test results
├── evaluation_report.json    # Comprehensive evaluation
└── plots/                    # Generated visualizations
    ├── training_rewards.png
    ├── verification_accuracy.png
    ├── verifier_success_rates.png
    └── verifier_processing_times.png
```

## Configuration

The scripts use the fast configuration by default. To modify settings:

1. Edit `src/config/training_config.py`
2. Modify the `get_fast_config()` function
3. Or create a custom configuration

## Example Output

### Training Data Generation

```
=== Data Preparation for RLVR Training ===

Using configuration: gpt2-medium
Max sequence length: 512
Batch size: 4

1. Generating synthetic training data...
Generated 1000 training examples

Sample training data:
Example 1:
Instruction: Write a function to calculate the factorial of a number.
Expected: def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)...
Type: code_generation, Difficulty: easy
```

### Verifier Testing

```
=== Verifier Development and Testing ===

Testing Code Verifier:

Test 1: Correct factorial function
  Result: CORRECT
  Score: 1.000
  Details: {'execution_success': True, 'output_match': True}

Code Verifier Summary: 3/4 tests passed
```

### Training Process

```
=== RLVR Training Pipeline ===

Using configuration: gpt2-medium
Learning rate: 1e-05
Batch size: 4
Number of episodes: 100

1. Loading training data...
Loaded 700 training examples and 150 validation examples

2. Initializing components...
✓ Initialized language model: gpt2-medium
✓ Initialized 3 verifiers
✓ Initialized hybrid reward function
```

### Evaluation Results

```
=== Evaluation Summary ===
Training Performance:
  total_episodes: 100
  average_reward: 0.723
  final_reward: 0.856
  reward_improvement: 0.133

Verification Performance:
  total_tests: 12
  code_success_rate: 0.750
  math_success_rate: 0.833
  logic_success_rate: 0.667

Recommendations:
  • Training shows good improvement in rewards
  • Logic verifier needs improvement (success rate: 66.67%)
```

## Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Make sure you're in the project root directory
   cd /path/to/ReinforcementLearningHybridReward

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Missing Data Files**

   ```bash
   # Run scripts in order
   python 01_data_preparation.py
   python 02_verifier_development.py
   python 03_rlvr_training.py
   python 04_evaluation.py
   ```

3. **Memory Issues**

   - Reduce batch size in configuration
   - Use smaller models
   - Reduce number of training examples

4. **Slow Training**
   - Use GPU if available
   - Reduce verification complexity
   - Use faster configurations

### Debug Mode

To enable debug logging:

```python
# In any script, change:
logger = setup_logging(log_level="INFO")

# To:
logger = setup_logging(log_level="DEBUG")
```

## Customization

### Adding New Task Types

1. Modify `generate_synthetic_data()` in `01_data_preparation.py`
2. Add new task templates
3. Create corresponding verifiers

### Adding New Verifiers

1. Create new verifier class inheriting from `BaseVerifier`
2. Add to verifier list in training scripts
3. Update verification testing

### Modifying Reward Function

1. Edit `HybridReward` configuration
2. Adjust weights for different components
3. Add new reward components

## Next Steps

After running the examples:

1. **Experiment with different configurations**

   - Try different learning rates
   - Adjust reward weights
   - Test different verifiers

2. **Add your own data**

   - Replace synthetic data with real examples
   - Create domain-specific verifiers
   - Customize reward functions

3. **Scale up**

   - Use larger models
   - Increase training data
   - Run distributed training

4. **Production deployment**
   - Optimize for performance
   - Add monitoring and logging
   - Implement error handling

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the implementation guide (`IMPLEMENTATION_GUIDE.md`)
3. Examine the source code in `src/`
4. Check generated logs and error messages
