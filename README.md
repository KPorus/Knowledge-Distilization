graph TD
    A[Start] --> B[Load and Preprocess Data]
    B --> C[Train Word2Vec Model]
    B --> D[Initialize Models]
    D --> E[Fine-Tune Teacher Model]
    D --> F[Initialize Student Model]
    E --> G[Evaluate Teacher on Validation Set]
    F --> H[Train Student with Knowledge Distillation]
    H --> I[Apply Pruning to Student Model]
    H --> J[Early Stopping Based on Validation Loss]
    G --> K{Evaluation Results}
    I --> K
    J --> K
    K --> L[Profile Student Model Efficiency]
    L --> M[Final Test Evaluation]
    M --> N[Stop]

    subgraph Data Preparation
        B --> C
    end

    subgraph Model Training
        D --> E
        D --> F
        E --> G
        F --> H
        H --> I
        H --> J
    end

    subgraph Evaluation
        K --> L
        L --> M
    end
