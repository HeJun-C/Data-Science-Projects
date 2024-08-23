import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from flask import Flask, request, jsonify, render_template
from collections import Counter
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Check for GPU availability
print("checking device...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize tokenizer and model
print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
print("loading derberta-v3-base...")
model = AutoModelForTokenClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=13).to(device)

# Load the model's state dictionary
model_save_path = 'model/model.pth'
print('loading pre-trained weights...')
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
print('weights loaded!')

# Personal identifiable information setup
pii_labels = ['B-email', 'B-id number', 'B-name', 'B-phone number',
              'B-address', 'B-personal url', 'B-user name',
              'I-id number', 'I-name', 'I-phone number',
              'I-address', 'I-personal url', 'O']

# Integer label to BIO format label mapping
pii_id2label = dict(enumerate(pii_labels))
# BIO format label to integer label mapping
pii_label2id = {v: k for k, v in pii_id2label.items()}

port = 8000
print(f"Starting Flask app on http://localhost:{port}")

def mask_pii(tokens, labels):
    masked_tokens = []
    for token, label in zip(tokens, labels):
        if label != 'O':
            # Extract the category (e.g., 'address' from 'B-address' or 'I-address')
            category = label[2:]
            # Define the class name dynamically
            class_name = f"{category}"
            # Create the masked token with the appropriate class
            masked_tokens.append(f"<span class='masked-text {class_name}'>&nbsp;{category}&nbsp;</span>")
        else:
            masked_tokens.append(token)
    return ' '.join(masked_tokens)

def pii_masker(example_input_text):
    # Tokenize the input text while keeping track of the original words
    tokenized_inputs = tokenizer(example_input_text, return_tensors="pt", truncation=True, padding='max_length', max_length=1024, is_split_into_words=False)
    example_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}

    # Extract the input IDs
    input_ids = example_inputs['input_ids'].squeeze().cpu().numpy()

    # Convert the input IDs back to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Decode the tokenized input to see special tokens
    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
    print('User input:', decoded_input)

    # Filter out special tokens
    special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    filtered_tokens = [token for token, token_id in zip(tokens, input_ids) if token_id not in special_tokens]

    # Set model to evaluation mode
    model.eval()

    print("Performing PII Detection...")
    # Perform prediction
    with torch.no_grad():
        outputs = model(input_ids=example_inputs['input_ids'], attention_mask=example_inputs['attention_mask'])

    # Extract logits
    logits = outputs.logits

    # Get the predicted class for each token
    predictions = torch.argmax(logits, dim=2).squeeze().cpu().numpy()

    # Filter out predictions for special tokens
    filtered_predictions = [pred for pred, token_id in zip(predictions, input_ids) if token_id not in special_tokens]

    # Function to join subwords and assign the most frequent label
    def join_subwords_and_labels(tokens, labels):
        joined_tokens = []
        joined_labels = []
        current_token = ""
        current_labels = []

        for token, label in zip(tokens, labels):
            if token.startswith("▁") or token.startswith("##"):
                if current_token:
                    joined_tokens.append(current_token)
                    joined_labels.append(Counter(current_labels).most_common(1)[0][0])
                current_token = token.replace("▁", "").replace("##", "")
                current_labels = [label]
            else:
                current_token += token
                current_labels.append(label)

        if current_token:
            joined_tokens.append(current_token)
            joined_labels.append(Counter(current_labels).most_common(1)[0][0])

        return joined_tokens, joined_labels

    # Join subwords and assign the most frequent labels
    joined_tokens, joined_labels = join_subwords_and_labels(filtered_tokens, filtered_predictions)

    # Convert joined labels to their corresponding tag names
    joined_labels = [pii_id2label[label] if label != -100 else "PAD" for label in joined_labels]

    masked_sentence = mask_pii(joined_tokens, joined_labels)

    print("Return masked:", masked_sentence)
    return masked_sentence

def save_user_data(input_text, output_text, file_path='data/stored_data.json'):
    # Create the data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Prepare the data to save
    data = {
        'input': input_text,
        'output': output_text
    }
    
    # Check if the file already exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
        existing_data.append(data)
    else:
        existing_data = [data]
    
    # Write the data to the file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    masked_sentence = pii_masker(user_input)
    save_user_data(user_input, masked_sentence)
    return jsonify(result=masked_sentence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)