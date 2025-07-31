#!/usr/bin/env python3


# =============================================================================
# SETUP (VSCode / Local Environment)

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# Use 'r' before the string to handle Windows paths correctly
FAKE_NEWS_PATH = r"C:\Users\Saber0710\Downloads\Fake.csv"
TRUE_NEWS_PATH = r"C:\Users\Saber0710\Downloads\True.csv"
OUTPUT_DIR = "processed_data" # A folder to save the output files

# Set up matplotlib for local environment
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Download required NLTK data
print("📚 Downloading NLTK data (if not already present)...")
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

print("✅ Setup complete!")
print("=" * 60)

# =============================================================================
# PHASE 1: DATASET LOADING (LOCAL VERSION)
# =============================================================================

def combine_datasets(fake_df, real_df):
    """Combine fake and real datasets"""
    # Add labels
    fake_df['label'] = 0  # FAKE = 0
    real_df['label'] = 1  # REAL = 1
    
    print(f"📰 Fake news dataset: {len(fake_df)} articles")
    print(f"📰 Real news dataset: {len(real_df)} articles")
    
    # Combine datasets
    df = pd.concat([fake_df, real_df], ignore_index=True)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"📊 Combined dataset: {len(df)} articles")
    print(f"📋 Dataset columns: {df.columns.tolist()}")
    
    return df

def load_data_from_local(fake_path, true_path):
    """Load datasets from local file paths."""
    print("🔧 PHASE 1: DATASET LOADING")
    print("=" * 40)
    print(f"📂 Attempting to load files:")
    print(f"  - Fake News: {fake_path}")
    print(f"  - True News: {true_path}")

    try:
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(true_path)
        print("✅ Successfully loaded datasets from local files.")
        return combine_datasets(fake_df, real_df)
    except FileNotFoundError as e:
        print(f"❌ ERROR: File not found. Please check your file paths.")
        print(f"  Details: {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading files: {e}")
        return None

# Load the dataset
df = load_data_from_local(FAKE_NEWS_PATH, TRUE_NEWS_PATH)

if df is not None:
    print("\n📊 DATASET INFORMATION:")
    print("=" * 40)
    print(f"Dataset shape: {df.shape}")
    
    print("\n🏷️ Label Distribution:")
    print(df['label'].value_counts())
    
    print("\n📝 Sample Data:")
    print(df.head(3))

print("=" * 60)

# =============================================================================
# PHASE 2: DATA PREPROCESSING
# =============================================================================

def preprocess_dataset(df):
    """Clean and preprocess the dataset."""
    print("🧹 PHASE 2: DATA PREPROCESSING")
    print("=" * 40)
    
    print(f"📊 Initial dataset size: {len(df)}")
    
    # 1. Remove duplicates and handle null values
    print("🔧 Removing duplicates and handling null values...")
    df_clean = df.drop_duplicates().reset_index(drop=True)
    print(f"   After removing duplicates: {len(df_clean)}")
    
    if 'title' in df_clean.columns:
        df_clean['title'] = df_clean['title'].fillna('')
    if 'text' in df_clean.columns:
        df_clean['text'] = df_clean['text'].fillna('')
    
    # Remove rows where both title and text are empty
    df_clean = df_clean[(df_clean['title'].str.len() > 0) | (df_clean['text'].str.len() > 0)]
    df_clean = df_clean.reset_index(drop=True)
    print(f"   After handling null values: {len(df_clean)}")
    
    # 2. Text preprocessing function
    def clean_text(text):
        """Clean and preprocess text"""
        if pd.isna(text) or text == '':
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_tokens = nltk.tokenize.word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words and len(word) > 1]
        
        return ' '.join(filtered_text)
    
    # Apply text cleaning
    print("🔤 Cleaning text data...")
    df_clean['title_clean'] = df_clean['title'].apply(clean_text) if 'title' in df_clean.columns else ''
    df_clean['text_clean'] = df_clean['text'].apply(clean_text) if 'text' in df_clean.columns else ''
    
    # 3. Create combined text feature
    print("🔗 Creating combined text feature...")
    df_clean['combined_text'] = (df_clean['title_clean'].astype(str) + ' ' + 
                                 df_clean['text_clean'].astype(str)).str.strip()
    
    # Remove rows with very short combined text
    df_clean = df_clean[df_clean['combined_text'].str.len() > 10].reset_index(drop=True)
    
    print(f"✅ Final dataset size: {len(df_clean)}")
    
    return df_clean

# Preprocess the dataset
if df is not None:
    df_processed = preprocess_dataset(df)
    
    print("\n📋 Processed Dataset Sample:")
    print(df_processed[['combined_text', 'label']].head(2))

print("=" * 60)

# =============================================================================
# DATA VISUALIZATION
# =============================================================================

def visualize_data(df):
    """Create visualizations for the dataset."""
    print("📊 DATA VISUALIZATION")
    print("=" * 40)
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fake News Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Label distribution
    label_counts = df['label'].value_counts()
    labels = ['FAKE', 'REAL']
    colors = ['#FF6B6B', '#4ECDC4']
    
    axes[0, 0].bar(labels, label_counts.values, color=colors, alpha=0.8)
    axes[0, 0].set_title('Distribution of News Labels', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Number of Articles')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Text length distribution
    text_lengths = df['combined_text'].str.len()
    axes[0, 1].hist(text_lengths, bins=50, alpha=0.7, color='#95A5A6', edgecolor='black')
    axes[0, 1].set_title('Distribution of Text Lengths (Characters)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Text Length (characters)')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Box plot of text length by label
    sns.boxplot(x='label', y=df['combined_text'].str.len(), data=df, ax=axes[1, 0], palette=colors)
    axes[1, 0].set_title('Text Length Distribution by Label', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticklabels(['FAKE', 'REAL'])
    axes[1, 0].set_xlabel('')
    axes[1, 0].set_ylabel('Text Length (characters)')

    # 4. Word count distribution
    word_counts = df['combined_text'].str.split().str.len()
    axes[1, 1].hist(word_counts, bins=50, alpha=0.7, color='#F39C12', edgecolor='black')
    axes[1, 1].set_title('Distribution of Word Counts', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Words')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Create visualizations
if df is not None:
    visualize_data(df_processed)

print("=" * 60)

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================

def split_dataset(df, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets."""
    print("🔄 TRAIN-TEST SPLIT")
    print("=" * 40)
    
    X = df['combined_text'].values
    y = df['label'].values
    
    print(f"📊 Splitting {len(X):,} samples...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Data Split Complete:")
    print(f"   🏋️ Training set: {len(X_train):,} samples")
    print(f"   🧪 Test set:     {len(X_test):,} samples")
    
    return X_train, X_test, y_train, y_test

# Split the dataset
if df is not None:
    X_train, X_test, y_train, y_test = split_dataset(df_processed)

print("=" * 60)

# =============================================================================
# SAVE PROCESSED DATA
# =============================================================================

def save_processed_data(df, X_train, X_test, y_train, y_test, output_dir):
    """Save processed dataframes to local CSV files."""
    print("💾 SAVING PROCESSED DATA")
    print("=" * 40)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📂 Output directory '{output_dir}' is ready.")
        
        full_path = os.path.join(output_dir, 'processed_news_dataset.csv')
        train_path = os.path.join(output_dir, 'train_data.csv')
        test_path = os.path.join(output_dir, 'test_data.csv')
        
        df.to_csv(full_path, index=False)
        pd.DataFrame({'text': X_train, 'label': y_train}).to_csv(train_path, index=False)
        pd.DataFrame({'text': X_test, 'label': y_test}).to_csv(test_path, index=False)
        
        print("✅ Files saved successfully:")
        print(f"   📄 {full_path}")
        print(f"   📄 {train_path}")
        print(f"   📄 {test_path}")

    except Exception as e:
        print(f"❌ Error saving files: {e}")

# Save processed data
if df is not None:
    save_processed_data(df_processed, X_train, X_test, y_train, y_test, OUTPUT_DIR)

print("=" * 60)

# =============================================================================
# FINAL SUMMARY AND NEXT STEPS
# =============================================================================

print("🎉 PHASE 1 & 2 COMPLETED SUCCESSFULLY!")
print("=" * 60)

if df is not None:
    print("✅ What we accomplished:")
    print("   📊 Loaded and preprocessed the dataset from local files")
    print("   🧹 Cleaned and combined text data")
    print("   📈 Generated and displayed data visualizations")
    print("   🔄 Split data into stratified train/test sets")
    print(f"  💾 Saved processed data to the '{OUTPUT_DIR}/' folder")
    
    print(f"\n📊 Final Dataset Summary:")
    print(f"   📰 Total articles: {len(df_processed):,}")
    print(f"   🏋️ Training samples: {len(X_train):,}")
    print(f"   🧪 Test samples:     {len(X_test):,}")
    
    print(f"\n🚀 Ready for Phase 3 (BERT Modeling):")
    print("   Your data is ready in these variables:")
    print("   - X_train, X_test: Preprocessed text data")
    print("   - y_train, y_test: Labels (0=FAKE, 1=REAL)")
    
else:
    print("❌ Script finished with errors. Dataset could not be loaded.")

print("\n🔗 Next: Use the generated CSV files or variables for Phase 3 to train your BERT model!")
print("=" * 60)