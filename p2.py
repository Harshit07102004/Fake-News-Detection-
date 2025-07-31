#!/usr/bin/env python3
"""
Fake News Detection using BERT - Phase 3: Model Building & Training (Local VSCode Version)
========================================================================================
This script implements BERT-based classification for fake news detection.
"""

# Import required libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR = "processed_data"
OUTPUT_DIR = "bert_model"

# Set up device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

print("=" * 60)

# =============================================================================
# PHASE 3: BERT MODEL SETUP
# =============================================================================

class FakeNewsDataset(Dataset):
    """Custom Dataset class for BERT tokenization"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def setup_bert_model():
    """Initialize BERT model and tokenizer"""
    print("🤖 PHASE 3: BERT MODEL SETUP")
    print("=" * 40)
    model_name = 'bert-base-uncased'
    print(f"📦 Loading pre-trained BERT model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print("✅ BERT model and tokenizer loaded successfully.")
    model = model.to(device)
    return model, tokenizer

def create_datasets(X_train, X_test, y_train, y_test, tokenizer, max_length=512):
    """Create PyTorch datasets for training and testing"""
    print("🔧 Creating tokenized datasets...")
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer, max_length)
    test_dataset = FakeNewsDataset(X_test, y_test, tokenizer, max_length)
    print(f"✅ Datasets created: {len(train_dataset)} train, {len(test_dataset)} test samples.")
    return train_dataset, test_dataset

# =============================================================================
# MODEL TRAINING SETUP
# =============================================================================

def compute_metrics(eval_pred):
    """Compute evaluation metrics for the Trainer"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

def setup_training_arguments(output_dir):
    """Setup training arguments"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Simplified arguments for compatibility
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=500,
        save_steps=1000,
        report_to=None,
        fp16=torch.cuda.is_available(),
    )
    print("⚙️ Training configuration set.")
    return training_args

def train_model(model, train_dataset, test_dataset, training_args):
    """Train the BERT model using HuggingFace Trainer"""
    print("🏋️ STARTING MODEL TRAINING")
    print("=" * 40)
    
    # Removed EarlyStoppingCallback for compatibility
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset, # Evaluation dataset is still needed for metrics
        compute_metrics=compute_metrics,
    )
    print("🚀 Training started...")
    trainer.train()
    print("✅ Training completed!")
    trainer.save_model()
    print(f"💾 Model saved to: {training_args.output_dir}")
    return trainer

# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(trainer, test_dataset, y_test):
    """Comprehensive model evaluation"""
    print("📊 MODEL EVALUATION")
    print("=" * 40)
    
    # Manually trigger evaluation if not done during training
    # This ensures you see the final performance metrics
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"📈 Final Evaluation Metrics: {eval_results}")

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    classes = ['FAKE', 'REAL']
    
    print("\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def run_phase3_pipeline(input_dir, output_dir):
    """Execute the complete Phase 3 pipeline"""
    print("🚀 STARTING PHASE 3: BERT MODEL TRAINING")
    print("=" * 60)
    
    try:
        train_data_path = os.path.join(input_dir, 'train_data.csv')
        test_data_path = os.path.join(input_dir, 'test_data.csv')
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        
        train_df.dropna(subset=['text', 'label'], inplace=True)
        test_df.dropna(subset=['text', 'label'], inplace=True)

        X_train = train_df['text'].values
        y_train = train_df['label'].values
        X_test = test_df['text'].values
        y_test = test_df['label'].values
        print(f"✅ Data loaded from '{input_dir}': {len(X_train)} train, {len(X_test)} test samples.")
    except FileNotFoundError:
        print(f"❌ ERROR: Preprocessed data not found in '{input_dir}'.")
        print("Please run the Phase 1 & 2 script first to generate the data.")
        return

    # Execute pipeline
    model, tokenizer = setup_bert_model()
    train_dataset, test_dataset = create_datasets(X_train, X_test, y_train, y_test, tokenizer, max_length=256)
    training_args = setup_training_arguments(output_dir)
    trainer = train_model(model, train_dataset, test_dataset, training_args)
    evaluate_model(trainer, test_dataset, y_test)
    
    print("\n🎉 PHASE 3 COMPLETED SUCCESSFULLY!")
    print(f"💾 Model and results saved to '{output_dir}'")

# Run the complete pipeline
if __name__ == "__main__":
    run_phase3_pipeline(INPUT_DIR, OUTPUT_DIR)