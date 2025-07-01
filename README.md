# Knowledge Distillation for Text Classification

## Introduction

This project demonstrates a knowledge distillation approach for text classification using the IMDB movie reviews dataset. The goal is to transfer knowledge from a large, high-performing teacher model (BERT) to a smaller, efficient student model (RNN with attention and Word2Vec embeddings), achieving competitive accuracy with reduced computational cost.

---

## Project Motivation

Large language models like BERT achieve state-of-the-art results but are computationally expensive and memory-intensive, making them impractical for deployment on resource-constrained devices. Knowledge distillation enables us to compress these models by training a smaller student model to mimic the behavior of a larger teacher, thus achieving a balance between performance and efficiency. This project aims to demonstrate the effectiveness of knowledge distillation and model pruning for real-world NLP applications.

---

## 3. Research Methodology

The methodology involves:
- Fine-tuning a BERT model as the teacher on the IMDB dataset.
- Training a custom RNN-based student model using a hybrid distillation loss that combines soft targets (teacher outputs), hard targets (true labels), and sequence loss.
- Applying model pruning to the student for further efficiency.
- Evaluating both models on accuracy, F1 score, and efficiency metrics.

---

## 3.1 Proposed Framework

- **Teacher Model:** BERT (Bidirectional Encoder Representations from Transformers) for sequence classification, fine-tuned on IMDB.
- **Student Model:** RNN (LSTM) with attention, initialized with Word2Vec embeddings, trained via knowledge distillation.
- **Distillation Loss:** Hybrid of KL divergence (soft targets), cross-entropy (hard targets), and sequence loss.
- **Pruning:** L1 unstructured pruning applied to LSTM and Linear layers in the student model.

---

## Hybrid Distillation Loss: Rationale and Details

The hybrid distillation loss combines three components:
- **Soft Loss (KL Divergence):** Encourages the student to match the teacher's output distribution (soft targets), capturing the teacher's knowledge about class similarities.
- **Hard Loss (Cross-Entropy):** Ensures the student learns from the true labels, maintaining task performance.
- **Sequence Loss:** Further aligns the student with the teacher's most confident predictions.

This combination allows the student to benefit from both the teacher's nuanced knowledge and the ground truth, leading to better generalization and robustness.

---

## 3.2 Data Set Analysis

- **Dataset:** IMDB movie reviews (50,000 labeled reviews: 25,000 positive, 25,000 negative).
- **Preprocessing:**
  - Tokenization using BERT tokenizer (max length 128).
  - Word2Vec embeddings trained on the training set.
- **Splits:**
  - Training: 80% of original train set
  - Validation: 20% of original train set
  - Test: Standard IMDB test set

| Class      | Count   |
|------------|---------|
| Positive   | 25,000  |
| Negative   | 25,000  |

---

### 3.2.1 Data Collection & Preprocessing

- **Data Collection:** Loaded using the `datasets` library.
- **Preprocessing Steps:**
  - Texts are tokenized with BERT tokenizer.
  - Word2Vec embeddings are trained on tokenized texts.
  - Data is formatted for PyTorch DataLoader.
  - Padding and truncation ensure uniform input size.

---

## 3.3 Algorithm/Model Analysis

### Teacher Model (BERT)
- Fine-tuned with two learning rates (lower for base, higher for classifier head).
- Adam optimizer and linear learning rate scheduler.
- Dropout added to classifier head to reduce overfitting.
- Evaluated on validation data for accuracy and F1 score.

### Student Model (RNN with Attention)
- Embedding layer initialized with Word2Vec.
- Two-layer LSTM with dropout.
- Attention mechanism to focus on important tokens.
- Fully connected output layer for classification.
- L1 pruning applied to LSTM and Linear layers for efficiency.

### Knowledge Distillation
- **Loss Function:** Hybrid of KL divergence (soft targets), cross-entropy (hard targets), and sequence loss.
- **Training:** Student learns from both teacher outputs and true labels.
- **Early Stopping:** Stops training if validation loss does not improve for 5 epochs.

---

## Model Efficiency and Pruning

Pruning is a technique to remove less important weights from a neural network, reducing its size and computational requirements without significantly impacting performance. In this project:
- **L1 Unstructured Pruning** is applied to the LSTM and Linear layers of the student model.
- This reduces the number of active parameters, making the model more suitable for deployment on edge devices.
- Efficiency is measured using PyTorch's profiler, reporting CPU and CUDA time, and parameter count.

---

## Evaluation Metrics

- **Accuracy:** Measures the proportion of correct predictions. Useful for balanced datasets.
- **F1 Score:** Harmonic mean of precision and recall, providing a better measure for imbalanced classes.
- **Efficiency Metrics:** Includes parameter count and CPU/CUDA time, important for real-world deployment.

These metrics provide a comprehensive view of both the effectiveness and efficiency of the models.

---

## 3.4 Explanation of Different Modules

- **RNNStudent:** Custom PyTorch module for the student model.
- **distillation_loss:** Combines soft and hard losses for distillation.
- **fine_tune_teacher:** Fine-tunes the BERT teacher model.
- **train_kd:** Trains the student model using knowledge distillation and pruning.
- **Data Preparation:** Tokenization, Word2Vec training, and DataLoader setup.

---

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install torch transformers datasets gensim scikit-learn
   ```

2. **Run the Notebook:**
   - Open `v7.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute all cells to train the teacher and student models, and evaluate performance.

3. **Results:**
   - Training/validation metrics for both teacher and student models.
   - Efficiency metrics and parameter count for the student model.
   - Final test accuracy and F1 score.

---

## Results

- **Teacher Model:**
  - Achieves high accuracy and F1 score on validation data after fine-tuning.
- **Student Model:**
  - Achieves competitive accuracy and F1 score with significantly fewer parameters and faster inference.
  - Example (from logs):
    - Student Model Parameter Count: ~3.9M
    - Final Test Accuracy: ~0.83
    - Final Test F1 Score: ~0.83
    - Efficiency metrics (CPU/CUDA time) are printed after training.

---

## Limitations and Future Work

- The student model, while efficient, may not capture all the nuances of the teacher, especially for more complex tasks.
- The current approach uses only LSTM-based students; exploring other architectures (e.g., CNN, GRU, Tiny Transformers) could yield better trade-offs.
- Further compression techniques like quantization or knowledge transfer from multiple teachers could be explored.
- Hyperparameter tuning and more advanced pruning strategies may improve results.
- The framework can be extended to multilingual or multi-label classification tasks.

---

## Additional Notes

- **Reproducibility:**
  - Set random seeds for torch, numpy, and other libraries if exact reproducibility is required.
  - Ensure GPU is available for faster training (the code will fall back to CPU if not).
- **Dependencies:**
  - Python 3.7+
  - torch, transformers, datasets, gensim, scikit-learn, numpy
- **Extensibility:**
  - The framework can be adapted for other text classification datasets.
  - The student model architecture can be modified (e.g., GRU, CNN, etc.).
  - Additional pruning or quantization techniques can be explored for further efficiency.
- **Troubleshooting:**
  - If you encounter `IProgress not found` warnings, update Jupyter and ipywidgets.
  - Ensure all dependencies are installed and compatible with your Python version.

---

## Acknowledgements

- HuggingFace Transformers and Datasets libraries for model and data loading.
- PyTorch for deep learning framework and pruning utilities.
- Gensim for Word2Vec implementation.
- Scikit-learn for evaluation metrics.
- The IMDB dataset creators for providing a benchmark for sentiment analysis.

---

## References
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index) 