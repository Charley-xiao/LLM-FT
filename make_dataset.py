# dataset.py
from datasets import load_dataset
from mindnlp.transformers import AutoTokenizer
import json

def load_and_preprocess_dataset(dataset_name, tokenizer_name, max_length=512, dataset_type='qa', qa_columns=['instruction', 'output']):
    dataset = load_dataset(dataset_name)

    print(f'Loaded dataset: {dataset_name}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f'Loaded tokenizer: {tokenizer_name}')

    def preprocess_function_text(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
    
    def preprocess_function_qa(examples):
        inputs = [q + " " + a for q, a in zip(examples[qa_columns[0]], examples[qa_columns[1]])]
        return tokenizer(inputs, padding='max_length', truncation=True, max_length=max_length)
    
    if dataset_type == 'text':
        tokenized_datasets = dataset.map(preprocess_function_text, batched=True)
    elif dataset_type == 'qa':
        tokenized_datasets = dataset.map(preprocess_function_qa, batched=True)
    else:
        raise ValueError("Unsupported dataset_type. Choose either 'text' or 'qa'.")
    
    if dataset_type == 'text':
        tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    elif dataset_type == 'qa':
        tokenized_datasets = tokenized_datasets.remove_columns([qa_columns[0], qa_columns[1]])
    
    tokenized_datasets.set_format('numpy')
    
    return tokenized_datasets
