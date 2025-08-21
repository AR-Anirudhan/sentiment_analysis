import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# Ensure necessary libraries are installed and updated
os.system('pip install --upgrade transformers datasets torch scikit-learn nltk accelerate tf-keras')

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.lower().split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(lemmatized_tokens)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

raw_datasets = load_dataset("imdb")

# Using a larger subset of the data for training and testing for high accuracy.
print("Using a larger dataset for higher accuracy...")
train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(15000))
test_dataset = raw_datasets["test"].shuffle(seed=42).select(range(5000))

def clean_dataset(dataset):
    cleaned_texts = [preprocess_text(text) for text in dataset['text']]
    return dataset.map(lambda example, idx: {'text': cleaned_texts[idx]}, with_indices=True)

print("Cleaning and tokenizing datasets... (This will take longer)")
cleaned_train_dataset = clean_dataset(train_dataset)
cleaned_test_dataset = clean_dataset(test_dataset)

tokenized_train_dataset = cleaned_train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = cleaned_test_dataset.map(tokenize_function, batched=True)
print("Dataset preparation complete.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results_high_accuracy',
    num_train_epochs=4,
    # Speed improvements for faster training on RTX cards
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    fp16=True,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_high_accuracy',
    logging_steps=50,
    eval_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

evaluation_results = trainer.evaluate()
print("\nFinal Evaluation Results:")
for key, value in evaluation_results.items():
    print(f"{key}: {value:.4f}")

output_model_dir = "./sentiment_model_distilbert_high_accuracy"
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)

print(f"\nModel and tokenizer saved to '{output_model_dir}'")
