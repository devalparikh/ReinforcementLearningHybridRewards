# RLVR Architecture & Process Flow

## High-Level System Architecture

```mermaid
graph TB
    subgraph "Data Pipeline"
        A[Training Data] --> B[Data Preprocessing]
        B --> C[Data Validation]
        C --> D[Train/Val/Test Split]
    end

    subgraph "Model & Generation"
        E[Language Model] --> F[Text Generation]
        F --> G[Model Output]
    end

    subgraph "Verification System"
        H[Code Verifier] --> I[Verification Results]
        J[Math Verifier] --> I
        K[Logic Verifier] --> I
        G --> H
        G --> J
        G --> K
    end

    subgraph "Reward Computation"
        I --> L[Hybrid Reward Function]
        L --> M[Reward Output]
    end

    subgraph "Training Loop"
        M --> N[PPO Trainer]
        N --> O[Policy Update]
        O --> E
    end

    subgraph "Monitoring & Logging"
        P[Metrics Tracker] --> Q[Training History]
        R[Structured Logging] --> S[Logs & Dashboards]
    end

    D --> E
    M --> P
    N --> R
```

## Detailed Component Architecture

```mermaid
graph LR
    subgraph "Configuration Layer"
        A1[TrainingConfig] --> A2[ModelConfig]
        A1 --> A3[VerifierConfig]
        A1 --> A4[RewardConfig]
    end

    subgraph "Core Components"
        B1[LanguageModel] --> B2[Policy]
        C1[BaseVerifier] --> C2[CodeVerifier]
        C1 --> C3[MathVerifier]
        C1 --> C4[LogicVerifier]
        D1[BaseReward] --> D2[HybridReward]
        D1 --> D3[RewardFactory]
    end

    subgraph "Training Pipeline"
        E1[RLVRTrainer] --> E2[PPOTrainer]
        E1 --> E3[ExperienceBuffer]
        E1 --> E4[MetricsTracker]
    end

    subgraph "Utilities"
        F1[Logging] --> F2[StructuredLogger]
        F1 --> F3[TrainingLogger]
        G1[Metrics] --> G2[PerformanceMonitor]
    end

    A1 --> B1
    A1 --> C1
    A1 --> D1
    A1 --> E1
```

## RLVR Training Process Flow

```mermaid
sequenceDiagram
    participant U as User
    participant T as RLVRTrainer
    participant LM as LanguageModel
    participant V as Verifiers
    participant R as RewardFunction
    participant P as PPOTrainer
    participant M as MetricsTracker

    U->>T: Start Training
    T->>T: Load Training Data

    loop For Each Episode
        T->>LM: Generate Response
        LM->>T: Model Output + Logprobs

        T->>V: Verify Output
        V->>T: Verification Results

        T->>R: Compute Reward
        R->>T: Reward Output

        T->>P: Update Policy
        P->>LM: Update Model Parameters

        T->>M: Log Metrics
        M->>T: Training Statistics
    end

    T->>U: Training Complete
```

## Data Flow Architecture

```mermaid
flowchart TD
    subgraph "Input Layer"
        A[Instructions] --> B[Context]
        A --> C[Expected Outputs]
    end

    subgraph "Processing Layer"
        D[Model Generation] --> E[Verification Pipeline]
        E --> F[Reward Computation]
        F --> G[Policy Update]
    end

    subgraph "Output Layer"
        H[Updated Model] --> I[Training Metrics]
        H --> J[Checkpoints]
        I --> K[Visualizations]
    end

    A --> D
    B --> D
    C --> E
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    H --> J
    I --> K
```

## Verification System Architecture

```mermaid
graph TB
    subgraph "Verification Types"
        A1[Code Verification] --> A2[Execute Code]
        A2 --> A3[Check Outputs]
        A3 --> A4[Safety Validation]

        B1[Math Verification] --> B2[Parse Expressions]
        B2 --> B3[Symbolic Computation]
        B3 --> B4[Numerical Validation]

        C1[Logic Verification] --> C2[Analyze Arguments]
        C2 --> C3[Check Fallacies]
        C3 --> C4[Coherence Analysis]
    end

    subgraph "Verification Results"
        D1[VerificationOutput] --> D2[Result Type]
        D1 --> D3[Score]
        D1 --> D4[Details]
        D1 --> D5[Confidence]
    end

    A4 --> D1
    B4 --> D1
    C4 --> D1
```

## Reward System Architecture

```mermaid
graph LR
    subgraph "Reward Components"
        A1[Verification Weight] --> A2[Verification Score]
        B1[Quality Weight] --> B2[Quality Metrics]
        C1[Diversity Weight] --> C2[Diversity Score]
        D1[Efficiency Weight] --> D2[Efficiency Metrics]
    end

    subgraph "Reward Computation"
        E1[Hybrid Reward] --> E2[Component Combination]
        E2 --> E3[Reward Shaping]
        E3 --> E4[Confidence Scoring]
    end

    subgraph "Reward Output"
        F1[RewardOutput] --> F2[Final Reward]
        F1 --> F3[Component Breakdown]
        F1 --> F4[Metadata]
        F1 --> F5[Confidence]
    end

    A2 --> E1
    B2 --> E1
    C2 --> E1
    D2 --> E1
    E4 --> F1
```

## Training Pipeline Architecture

```mermaid
graph TB
    subgraph "Data Management"
        A1[Dataset Loading] --> A2[Data Preprocessing]
        A2 --> A3[Data Validation]
        A3 --> A4[Batch Generation]
    end

    subgraph "Training Loop"
        B1[Episode Generation] --> B2[Model Inference]
        B2 --> B3[Verification]
        B3 --> B4[Reward Computation]
        B4 --> B5[Policy Update]
        B5 --> B6[Experience Storage]
    end

    subgraph "Monitoring"
        C1[Metrics Collection] --> C2[Performance Tracking]
        C2 --> C3[Logging]
        C3 --> C4[Visualization]
    end

    A4 --> B1
    B6 --> C1
    B5 --> B1
```

## Component Dependencies

```mermaid
graph TD
    A[RLVRTrainer] --> B[LanguageModel]
    A --> C[Verifiers]
    A --> D[RewardFunction]
    A --> E[PPOTrainer]
    A --> F[MetricsTracker]

    B --> G[Policy]
    C --> H[BaseVerifier]
    D --> I[BaseReward]
    E --> J[ExperienceBuffer]
    F --> K[Logging]

    L[TrainingConfig] --> A
    L --> B
    L --> C
    L --> D
    L --> E
```

## File Structure Overview

```mermaid
graph TD
    subgraph "Root Directory"
        A[README.md] --> B[requirements.txt]
        A --> C[setup.py]
        A --> D[IMPLEMENTATION_GUIDE.md]
    end

    subgraph "Source Code"
        E[src/] --> F[config/]
        E --> G[data/]
        E --> H[verifiers/]
        E --> I[rewards/]
        E --> J[models/]
        E --> K[training/]
        E --> L[utils/]
    end

    subgraph "Notebooks"
        M[notebooks/] --> N[01_data_preparation.ipynb]
        M --> O[02_verifier_development.ipynb]
        M --> P[03_rlvr_training.ipynb]
        M --> Q[04_evaluation.ipynb]
    end

    subgraph "Key Files"
        F --> R[training_config.py]
        G --> S[dataset.py]
        G --> T[preprocessing.py]
        H --> U[base_verifier.py]
        H --> V[code_verifier.py]
        H --> W[math_verifier.py]
        H --> X[logic_verifier.py]
        I --> Y[base_reward.py]
        I --> Z[hybrid_reward.py]
        I --> AA[reward_factory.py]
        J --> BB[language_model.py]
        J --> CC[policy.py]
        K --> DD[rlvr_trainer.py]
        K --> EE[ppo_trainer.py]
        K --> FF[experience_buffer.py]
        L --> GG[logging.py]
        L --> HH[metrics.py]
    end
```

## Training Workflow

```mermaid
stateDiagram-v2
    [*] --> DataPreparation
    DataPreparation --> VerifierDevelopment
    VerifierDevelopment --> ModelInitialization
    ModelInitialization --> TrainingLoop

    state TrainingLoop {
        [*] --> GenerateResponse
        GenerateResponse --> VerifyOutput
        VerifyOutput --> ComputeReward
        ComputeReward --> UpdatePolicy
        UpdatePolicy --> LogMetrics
        LogMetrics --> CheckConvergence
        CheckConvergence --> GenerateResponse : Continue
        CheckConvergence --> [*] : Converged
    }

    TrainingLoop --> Evaluation
    Evaluation --> [*]
```

## Key Features Summary

```mermaid
mindmap
  root((RLVR System))
    Modular Architecture
      Extensible Verifiers
      Configurable Rewards
      Pluggable Models
    Production Ready
      Comprehensive Logging
      Error Handling
      Configuration Validation
      Metrics Tracking
    Verification Types
      Code Execution
      Mathematical Verification
      Logical Reasoning
      Custom Verifiers
    Reward System
      Hybrid Rewards
      Configurable Weights
      Reward Shaping
      Confidence Scoring
    Training Pipeline
      PPO Algorithm
      Experience Buffer
      Policy Updates
      Checkpointing
    Monitoring
      Real-time Metrics
      Visualization
      Performance Tracking
      Debugging Tools
```
