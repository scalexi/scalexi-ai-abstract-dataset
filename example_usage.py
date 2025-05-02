#!/usr/bin/env python3
"""
ScaleXI AI Abstract Dataset - Example Usage

This script demonstrates how to load and explore the ScaleXI AI Abstract Dataset,
as well as implement a simple classification model to detect AI-generated academic abstracts.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# Optional dependencies for the advanced examples (will be skipped if not installed)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def main():
    """Main function to demonstrate dataset usage"""
    print("ScaleXI AI Abstract Dataset - Example Usage")
    print("-" * 50)
    
    # 1. Load the dataset
    print("\n1. Loading the ScaleXI AI Abstract Dataset")
    data = []
    with open('ai_detection_dataset.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"ScaleXI dataset loaded with {len(df)} academic abstracts")
    
    # 2. Explore the dataset
    print("\n2. Dataset statistics")
    ai_count = df['ai_generated'].sum()
    human_count = len(df) - ai_count
    
    print(f"AI-generated academic abstracts: {ai_count} ({ai_count/len(df)*100:.2f}%)")
    print(f"Human-written academic abstracts: {human_count} ({human_count/len(df)*100:.2f}%)")
    
    # 3. Prepare data for classification
    print("\n3. Preparing academic abstracts for classification")
    # Handle class imbalance by undersampling the majority class
    ai_abstracts = df[df['ai_generated'] == 1]
    human_abstracts = df[df['ai_generated'] == 0]
    
    # Sample human abstracts to be 3x the AI abstracts
    human_sample = human_abstracts.sample(n=len(ai_abstracts) * 3, random_state=42)
    balanced_data = pd.concat([ai_abstracts, human_sample])
    
    # Shuffle the data
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into training and testing sets (80% train, 20% test)
    train_data, test_data = train_test_split(
        balanced_data, 
        test_size=0.2, 
        random_state=42, 
        stratify=balanced_data['ai_generated']
    )
    
    print(f"Training set size: {len(train_data)} academic abstracts")
    print(f"Testing set size: {len(test_data)} academic abstracts")
    
    # Create feature vectors using TF-IDF
    print("\nExtracting features from academic abstract text")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train_data['abstract'])
    X_test = vectorizer.transform(test_data['abstract'])
    
    y_train = train_data['ai_generated']
    y_test = test_data['ai_generated']
    
    print(f"Features extracted. Shape of training features: {X_train.shape}")
    
    # 4. Build a simple classifier
    print("\n4. Training academic abstract classifier model")
    # Train a Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # 5. Evaluate the model
    print("\n5. Evaluating academic abstract classification model")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Feature importance
    print("\n6. Top 10 most important features for academic abstract classification")
    # Get feature importance from the Random Forest model
    feature_importance = rf_model.feature_importances_
    features = vectorizer.get_feature_names_out()
    
    # Create a DataFrame for easier visualization
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    })
    
    # Sort by importance and show top 10
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    print(importance_df.head(10))
    
    # 7. Test with custom example
    print("\n7. Testing with a custom academic abstract example")
    test_abstract = """This paper introduces a novel approach to quantum computing that leverages 
    entanglement properties of qubits to solve complex optimization problems. We demonstrate 
    a 30% improvement in computational efficiency compared to classical methods."""
    
    # Convert text to feature vector
    features = vectorizer.transform([test_abstract])
    
    # Get prediction and probability
    prediction = rf_model.predict(features)[0]
    probability = rf_model.predict_proba(features)[0][1]  # Probability of being AI-generated
    
    # Return result
    result = "AI-generated" if prediction == 1 else "Human-written"
    print(f"The model predicts this academic abstract is: {result}")
    print(f"Probability of being AI-generated: {probability:.4f} ({probability*100:.2f}%)")
    
    # 8. Demonstrating Core Challenge 2: Humanizing AI Text
    if ADVANCED_AVAILABLE:
        print("\n8. Core Challenge 2 Demo: Humanizing AI-Generated Academic Abstracts")
        try:
            # Get a sample AI-generated abstract
            ai_abstract = ai_abstracts.iloc[0]['abstract']
            print("\nOriginal AI-generated Academic Abstract:")
            print(ai_abstract)
            
            # Simple rule-based humanization (for demonstration only)
            # In a real solution, you would use a fine-tuned model on the ScaleXI dataset
            humanized_abstract = simple_humanize(ai_abstract)
            print("\nHumanized Academic Abstract Version (simple rules):")
            print(humanized_abstract)
            
            # Demonstrate evaluation
            print("\nEvaluation:")
            # 1. Check if still detected as AI
            features = vectorizer.transform([humanized_abstract])
            ai_prob_before = rf_model.predict_proba(features)[0][1]
            features = vectorizer.transform([ai_abstract])
            ai_prob_after = rf_model.predict_proba(features)[0][1]
            print(f"AI detection probability (original): {ai_prob_before:.4f}")
            print(f"AI detection probability (humanized): {ai_prob_after:.4f}")
            
            # 2. Measure semantic similarity (would use better metrics in a real solution)
            if 'nltk' in globals():
                try:
                    reference = word_tokenize(ai_abstract.lower())
                    candidate = word_tokenize(humanized_abstract.lower())
                    bleu_score = sentence_bleu([reference], candidate)
                    print(f"BLEU score (academic content preservation): {bleu_score:.4f}")
                except:
                    print("Could not compute BLEU score (NLTK resources may be missing)")
        except:
            print("Skipping academic abstract humanization demo due to errors")
    else:
        print("\n8. Core Challenge 2: Humanizing AI-Generated Academic Abstracts (skipped, requires additional packages)")
        print("Install transformers, torch, and nltk to see this demo")
    
    print("\nExplore the ScaleXI AI Abstract Dataset further and develop your own approaches for academic text classification!")


def simple_humanize(text):
    """
    A very simple rule-based humanization function for demonstration purposes.
    This is NOT an effective solution but shows the concept of transforming
    AI-generated academic abstracts into more human-like academic writing.
    A real solution would use a fine-tuned language model on academic writing patterns.
    """
    # Split into sentences 
    sentences = text.replace("\n", " ").split(". ")
    
    # Simple rules tailored for academic abstracts (these are just for demo, not effective)
    humanized = []
    for i, sentence in enumerate(sentences):
        if not sentence:
            continue
            
        # Add some "human-like" academic writing variations
        if i == 0:
            # First sentence more academic sometimes
            sentence = sentence.replace("This paper ", "In this research, we ")
            sentence = sentence.replace("We present ", "Our study explores ")
        
        # Add academic transition words occasionally
        if i % 3 == 0 and len(sentence) > 10:
            words = sentence.split()
            if len(words) > 5:
                # Insert academic transition words
                academic_words = ["moreover", "furthermore", "consequently", "specifically"]
                idx = min(3, len(words)-1)
                words.insert(idx, np.random.choice(academic_words))
                sentence = " ".join(words)
        
        # Vary sentence structure sometimes for academic writing
        if i % 2 == 1 and sentence.startswith("We ") and len(sentence) > 15:
            sentence = sentence[3:]  # Remove "We "
            sentence = "Our analysis " + sentence
            
        humanized.append(sentence)
    
    # Join back with periods and occasional paragraph breaks like in academic writing
    result = ". ".join(humanized)
    if len(result) > 200:
        # Add a paragraph break near the middle (common in academic abstracts)
        mid = len(result) // 2
        idx = result.find(". ", mid)
        if idx > 0:
            result = result[:idx+1] + "\n\n" + result[idx+1:]
            
    return result + ("." if not result.endswith(".") else "")


if __name__ == "__main__":
    main() 