import json
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset

# Load the JSON file
with open('/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/Packages/Transcrition/Transcrition_Tiny.json', 'r') as f:
    data = json.load(f)

# Extract the text values
texts = list(data.values())[20001:]  # Limit to 500 texts

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define batch size
batch_size = 50


# Tokenize the text in batches
def tokenize_batch(batch_texts):
    return tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)


# Process the text in batches
all_pooled_outputs = []
processed_texts = 0

total_texts = len(texts)

for i in range(0, total_texts, batch_size):
    batch_texts = texts[i:i + batch_size]
    inputs = tokenize_batch(batch_texts)

    with torch.no_grad():
        outputs = model(**inputs)

    pooled_output = outputs.pooler_output
    all_pooled_outputs.append(pooled_output)

    processed_texts += len(batch_texts)
    percentage = (processed_texts / total_texts) * 100
    print(f'Processing: {percentage:.2f}% completed')

# Concatenate all pooled outputs
total_pooled_output = torch.cat(all_pooled_outputs, dim=0)

# Save the features to a file
torch.save(total_pooled_output, 'bert_features_5.pt')
