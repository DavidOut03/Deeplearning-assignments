from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

def plot_training_history(history, model_name):
    """Plot training and validation loss/accuracy."""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def calculate_bleu_score(reference, hypothesis):
    """Calculate BLEU score for a translation."""
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)

def evaluate_translations(model, test_sentences, vectorizer, preprocess_func, translate_func):
    """Evaluate model translations."""
    results = []
    for sentence in test_sentences:
        translated = translate_func(sentence, model, vectorizer)
        bleu_score = calculate_bleu_score(sentence, translated)
        results.append({
            'input': sentence,
            'translation': translated,
            'bleu_score': bleu_score,
            'input_length': len(sentence.split()),
            'output_length': len(translated.split())
        })
    return pd.DataFrame(results)

def analyze_length_errors(results_df):
    """Analyze length-based errors in translations."""
    plt.figure(figsize=(12, 5))
    
    # Plot length distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=results_df, x='input_length', bins=20)
    plt.title('Input Length Distribution')
    plt.xlabel('Input Length')
    plt.ylabel('Frequency')
    
    # Plot length differences
    plt.subplot(1, 2, 2)
    results_df['length_diff'] = results_df['output_length'] - results_df['input_length']
    sns.histplot(data=results_df, x='length_diff', bins=20)
    plt.title('Length Difference Distribution')
    plt.xlabel('Output Length - Input Length')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def compare_models(rnn_results, transformer_results, rnn_training_time, transformer_training_time, rnn_inference_time, transformer_inference_time):
    """Compare RNN and Transformer models."""
    comparison = pd.DataFrame({
        'Metric': ['Average BLEU Score', 'Average Length Difference', 'Training Time', 'Inference Time'],
        'RNN': [
            rnn_results['bleu_score'].mean(),
            rnn_results['length_diff'].abs().mean(),
            rnn_training_time,
            rnn_inference_time
        ],
        'Transformer': [
            transformer_results['bleu_score'].mean(),
            transformer_results['length_diff'].abs().mean(),
            transformer_training_time,
            transformer_inference_time
        ]
    })
    
    plt.figure(figsize=(12, 6))
    comparison.set_index('Metric').plot(kind='bar')
    plt.title('Model Comparison')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return comparison

def analyze_error_patterns(results_df):
    """Analyze common error patterns in translations."""
    patterns = {
        'word_order': 0,
        'missing_words': 0,
        'extra_words': 0,
        'incorrect_translation': 0
    }
    
    for idx, row in results_df.iterrows():
        if row['length_diff'] < 0:
            patterns['missing_words'] += 1
        elif row['length_diff'] > 0:
            patterns['extra_words'] += 1
        
        # Add more sophisticated pattern analysis here
    
    plt.figure(figsize=(10, 6))
    plt.bar(patterns.keys(), patterns.values())
    plt.title('Distribution of Error Types')
    plt.xlabel('Error Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return patterns