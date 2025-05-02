# ScaleXI AI Abstract Dataset

A comprehensive dataset for AI-generated text detection in academic abstracts.

## Dataset Overview

This repository contains a dataset designed for training and evaluating AI text detection models, particularly focused on academic abstracts. The dataset includes both human-written and AI-generated abstracts, providing a valuable resource for researchers working on AI text classification and detection challenges.

### Dataset Statistics

- Total abstracts: 2,010
- Human-written abstracts: 1,976 (98.3%)
- AI-generated abstracts: 34 (1.7%)
- Sources: arXiv papers and AI-generated content

## Dataset Structure

The dataset is available in two formats:

- `ai_detection_dataset.csv`: CSV format
- `ai_detection_dataset.jsonl`: JSON Lines format (recommended for easier parsing)

### Schema

Each entry in the dataset contains the following fields:

- `source`: Origin of the abstract (e.g., "arxiv", "ai_gpt")
- `title`: Title of the paper
- `abstract`: The abstract text
- `authors`: List of authors (comma-separated in CSV, array in JSONL)
- `published_date`: Date of publication
- `categories`: Academic categories/field
- `doi`: Digital Object Identifier (may be null for AI-generated content)
- `url`: URL to the original paper (may be null for AI-generated content)
- `paper_id`: Unique identifier for the paper
- `word_count`: Number of words in the abstract
- `journal`: Journal where the paper was published (may be null)
- `pubmed_id`: PubMed identifier (may be null)
- `ai_generated`: Boolean indicating if the abstract was AI-generated (0 = human, 1 = AI)
- `metadata`: Additional metadata (JSONL format only)

## Usage

### Loading the Data

#### Python with Pandas (CSV)
```python
import pandas as pd

# Load CSV
data = pd.read_csv('ai_detection_dataset.csv')

# Access AI-generated abstracts
ai_abstracts = data[data['ai_generated'] == 1]

# Access human-written abstracts
human_abstracts = data[data['ai_generated'] == 0]
```

#### Python with JSON Lines (Recommended)
```python
import json

# Load JSONL
data = []
with open('ai_detection_dataset.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Convert to DataFrame if needed
import pandas as pd
df = pd.DataFrame(data)

# Access AI-generated abstracts
ai_abstracts = df[df['ai_generated'] == 1]

# Access human-written abstracts
human_abstracts = df[df['ai_generated'] == 0]
```

### Data Splitting

For machine learning tasks, you might want to create training, validation, and test sets:

```python
from sklearn.model_selection import train_test_split

# Create a balanced dataset with equal numbers of AI and human abstracts
human_sample = human_abstracts.sample(n=len(ai_abstracts) * 3, random_state=42)
balanced_data = pd.concat([ai_abstracts, human_sample])

# Split into train, validation, and test sets
train_data, temp_data = train_test_split(balanced_data, test_size=0.3, random_state=42, stratify=balanced_data['ai_generated'])
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['ai_generated'])
```

## ScaleXI AI Text Challenges

All challenges below are designed to be solved using the ScaleXI AI Abstract Dataset as the primary source data. Each challenge focuses on different aspects of academic abstract analysis and AI text detection.

### Core Challenge 1: Fine-tuning LLMs for Academic Abstract Classification

Develop a state-of-the-art AI text detector by fine-tuning large language models (open-source or otherwise) specifically on the ScaleXI AI Abstract Dataset. The goal is to create the most accurate classifier that can distinguish between human-written and AI-generated academic abstracts in this dataset.

**Approach:**
1. Choose an open-source LLM foundation model (e.g., Llama 3, Mistral, Falcon)
2. Fine-tune the model on the ScaleXI AI Abstract Dataset using techniques like PEFT
3. Optimize using LoRA, QLoRA, or other parameter-efficient methods
4. Evaluate performance on a held-out test set from this dataset

**Example starter code:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

# Load the ScaleXI AI Abstract Dataset
data = []
with open('ai_detection_dataset.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# Prepare data for training
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# Load base model
model_name = "mistralai/Mistral-7B-v0.1"  # or any other suitable base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
```

### Core Challenge 2: Humanizing AI-Generated Academic Abstracts

Train a language model to transform AI-generated abstracts from the ScaleXI dataset into more human-like writing that preserves the academic content and meaning. The goal is to develop a system that can "humanize" AI-generated abstracts from this dataset to make them less detectable by AI text classifiers.

**Approach:**
1. Analyze the linguistic differences between human and AI-written abstracts in the ScaleXI dataset
2. Create paired examples using the dataset's AI-generated abstracts
3. Fine-tune a model to transform AI text patterns into more human-like academic writing patterns
4. Evaluate using both automated metrics and classifiers trained on the same dataset

**Evaluation criteria:**
- Preservation of original academic content and information from the abstract
- Reduction in detectability by classifiers trained on the ScaleXI dataset
- Natural flow and academic writing style consistency
- Maintaining domain-specific terminology from the original abstract

**Complete example for model training and usage:**
```python
import torch
import pandas as pd
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset

# Step 1: Load the ScaleXI dataset
data = []
with open('ai_detection_dataset.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# Step 2: Get AI and human abstracts
ai_abstracts = df[df['ai_generated'] == 1]['abstract'].tolist()
human_abstracts = df[df['ai_generated'] == 0]['abstract'].tolist()

# Step 3: Create a paired dataset for training
# There are two possible approaches:

# Approach A: Human → AI (Recommended for larger training datasets)
# This approach uses human abstracts as input and generates AI-style abstracts as output
# We can use this direction and then reverse it during inference

print("Using Human → AI approach for more training data")

# Since we have more human abstracts than AI abstracts, we can use an existing
# LLM to generate AI-style versions of human abstracts
from transformers import pipeline

# Load a text generation model
generator = pipeline('text-generation', model='gpt2')

# Function to create an "AI-style" version of a human abstract
def create_ai_style_abstract(human_abstract, max_length=512):
    # In a real implementation, you would use a more sophisticated method
    # For example, you could fine-tune a model to mimic AI-generated abstracts
    # This is a placeholder using a simple prompt technique
    prompt = f"Generate an academic abstract in AI writing style based on: {human_abstract[:100]}..."
    
    # Generate AI-style text
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    ai_style_text = result[0]['generated_text']
    
    # Extract just the generated abstract part (not the prompt)
    if prompt in ai_style_text:
        ai_style_text = ai_style_text[len(prompt):].strip()
    
    return ai_style_text

# Create paired examples (human → AI)
human_to_ai_data = []

# Use a subset of human abstracts to keep the example manageable
for human_abstract in human_abstracts[:100]:  # Adjust number as needed
    ai_style_abstract = create_ai_style_abstract(human_abstract)
    human_to_ai_data.append({
        "human_text": human_abstract,
        "ai_text": ai_style_abstract
    })

# For inference, we'll reverse the transformation direction

# Approach B: AI → Human (Original approach)
# This directly uses our limited set of AI abstracts as source
print("\nAlternative: Using AI → Human approach with existing dataset")

# Function to find similar human abstracts
def find_similar_human_abstract(ai_abstract, human_abstracts, n=5):
    # In a real implementation, use embedding similarity or better matching
    # This is just a placeholder for demonstration
    words = set(ai_abstract.lower().split())
    scores = []
    
    for human_abstract in human_abstracts:
        human_words = set(human_abstract.lower().split())
        overlap = len(words.intersection(human_words))
        scores.append((overlap, human_abstract))
    
    # Return the most similar human abstract
    scores.sort(reverse=True)
    return scores[0][1] if scores else human_abstracts[0]

# Create paired examples (AI → human)
ai_to_human_data = []
for ai_abstract in ai_abstracts:
    human_abstract = find_similar_human_abstract(ai_abstract, human_abstracts)
    ai_to_human_data.append({
        "ai_text": ai_abstract,
        "human_text": human_abstract
    })

# Choose which approach to use
# Approach A gives more training data but may introduce noise
# Approach B uses real AI abstracts but provides less training data
use_human_to_ai_approach = True  # Set to False to use approach B

paired_data = human_to_ai_data if use_human_to_ai_approach else ai_to_human_data
print(f"Created {len(paired_data)} paired examples for training")

# For human → AI approach, we'll train in that direction but reverse at inference time
train_input_key = "human_text" if use_human_to_ai_approach else "ai_text"
train_target_key = "ai_text" if use_human_to_ai_approach else "human_text"

# Step 4: Prepare dataset for fine-tuning
train_dataset = Dataset.from_pandas(pd.DataFrame(paired_data))

# Step 5: Load base model for fine-tuning
# We'll use T5 as a base model for sequence-to-sequence transformation
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Step 6: Prepare the dataset
def preprocess_function(examples):
    # The prompt depends on which approach we're using
    if use_human_to_ai_approach:
        # For human → AI, we're learning to convert human text to AI style
        inputs = ["convert-to-ai-style: " + text for text in examples[train_input_key]]
    else:
        # For AI → human, we're learning to humanize AI text
        inputs = ["humanize: " + text for text in examples[train_input_key]]
    
    targets = examples[train_target_key]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Process datasets
tokenized_datasets = train_dataset.map(preprocess_function, batched=True)

# Step 7: Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./humanizer-model",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
)

# Prepare data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Step 8: Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Step 9: Fine-tune the model
trainer.train()

# Step 10: Save the model
model.save_pretrained("./humanizer-model")
tokenizer.save_pretrained("./humanizer-model")

# Step 11: Function to humanize text using the fine-tuned model
def humanize_abstract(ai_abstract, model_path="./humanizer-model"):
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Prepare input based on which training approach was used
    if use_human_to_ai_approach:
        # If we trained human → AI, we need to reverse the process
        # This means using the model to predict AI text from human text,
        # but we'll use it in reverse: input AI text and keep only the encoder
        # or use a different technique like style transfer
        
        # This is a simplified approach for demonstration:
        # We could use the model to get embeddings for AI text features
        # and then generate human-like text with different decoding parameters
        input_text = "convert-from-ai-style: " + ai_abstract
    else:
        # If we trained AI → human directly, we can use the model as is
        input_text = "humanize: " + ai_abstract
        
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate humanized text
    # For the human → AI approach, we might use different generation parameters
    # to reverse the transformation direction
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=512,
        num_beams=4,
        no_repeat_ngram_size=3,
        temperature=0.7 if use_human_to_ai_approach else 0.9,  # Higher temperature for reverse direction
    )
    humanized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return humanized_text

# Example usage
test_ai_abstract = ai_abstracts[0]
print("Original AI-generated abstract:")
print(test_ai_abstract)
print("\nHumanized abstract:")
print(humanize_abstract(test_ai_abstract))
```

This example shows the complete process of:
1. Loading and analyzing the ScaleXI dataset
2. Creating paired examples for training
3. Fine-tuning a sequence-to-sequence model (T5)
4. Using the fine-tuned model to humanize new AI-generated abstracts

Note that a real implementation would require more sophisticated methods for creating paired examples and evaluating results with the specialized metrics mentioned in the evaluation criteria.

### Additional Challenge 1: Academic Abstract Classification

Create a traditional machine learning model that can accurately classify abstracts in the ScaleXI dataset as either AI-generated or human-written academic content.

**Approach:**
1. Extract text features from the abstracts in the dataset
2. Train various classifiers (Random Forest, SVM, etc.) on these features
3. Evaluate performance specifically on academic abstracts from this domain

**Metrics to consider:**
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

**Example approach:**
```python
# Load the ScaleXI dataset
data = []
with open('ai_detection_dataset.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_data['abstract'])
X_test = vectorizer.transform(test_data['abstract'])

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, train_data['ai_generated'])

# Evaluation
predictions = model.predict(X_test)
print(classification_report(test_data['ai_generated'], predictions))
```

### Additional Challenge 2: Academic Domain Generalization

Using the ScaleXI dataset's diverse academic fields, train models on abstracts from certain academic domains and test their ability to detect AI-generated text in unfamiliar academic domains.

**Approach:**
1. Group abstracts in the dataset by academic categories/fields
2. Train detection models on specific academic domains (e.g., computer science, physics)
3. Evaluate how well they transfer to detecting AI-generated text in other academic domains

### Additional Challenge 3: Academic Writing Feature Analysis

Identify linguistic features that differentiate AI-generated from human-written academic abstracts specifically in the ScaleXI dataset.

**Approach:**
1. Compare lexical and syntactic patterns between AI and human abstracts in the dataset
2. Analyze domain-specific academic terminology usage
3. Examine citation patterns and research claim structures

**Potential features to extract from the dataset:**
- Lexical diversity in academic writing (type-token ratio)
- Academic syntactic complexity (parse tree depth)
- Technical term density and usage patterns
- Citation and reference patterns
- Academic discourse markers

### Additional Challenge 4: Robustness to Academic Paraphrasing

Test how well detection models trained on the ScaleXI dataset perform when academic abstracts undergo paraphrasing that maintains the scholarly content.

**Methodology:**
1. Create paraphrased versions of academic abstracts from the dataset
2. Evaluate detector performance on original vs. paraphrased academic texts
3. Analyze which academic writing features are most preserved or altered

### Additional Challenge 5: Few-Shot Academic Abstract Detection

Develop models that can detect AI-generated academic texts with minimal training examples from the ScaleXI dataset.

**Experiment:**
1. Train with varying small percentages of the dataset (1%, 5%, 10%, etc.)
2. Evaluate how effectively models can learn to identify AI-generated academic writing
3. Compare performance across different academic domains with limited examples

### Additional Challenge 6: Handling Class Imbalance in Academic Text Detection

The ScaleXI dataset has a significant class imbalance (1.7% AI-generated vs. 98.3% human-written). Develop strategies to handle this imbalance effectively for academic abstract classification.

**Techniques to explore with this dataset:**
- Oversampling AI-generated academic abstracts
- Undersampling human-written academic abstracts
- Synthetic academic abstract generation
- Cost-sensitive learning for academic classification
- Ensemble methods optimized for imbalanced academic text detection

## Citation

If you use this dataset in your research, please cite:

```
@dataset{scalexi2023aiabstract,
  title = {ScaleXI AI Abstract Dataset},
  author = {ScaleXI},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/scalexi/scalexi-ai-abstract-dataset}
}
```

## License

[MIT License](LICENSE)

## Contact

For questions or feedback about the dataset, please open an issue in this repository. 