"""
main.py

This script trains a custom AI chat bot using Meta's LLaMA model.
It includes data loading, preprocessing, and training steps.

Requirements:
- torch
- transformers
- pandas
- nltk

Usage:
python main.py --data_path path_to_your_dataset.csv
"""

import re
import argparse

import pandas as pd

from nltk.tokenize import word_tokenize
from transformers import (
    LLaMAForCausalLM,
    LLaMATokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def preprocess_text(text):
    """
    Preprocess the input text by lowercasing, removing special characters,
    and tokenizing.

    Args:
        text (str): The text to preprocess.

    Returns:
        List[str]: The tokenized text.
    """
    # Lowercase the text
    text = text.lower()
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Tokenize the text
    tokens = word_tokenize(text)
    return tokens


def load_data(data_path):
    """
    Loads and pre processes the dataset from the specified path.

    Args:
        data_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: The loaded and preprocessed dataset.
    """
    # Load the dataset
    data = pd.read_csv(data_path)
    # Apply preprocessing
    data["text"] = data["text"].apply(preprocess_text)
    return data


def tokenize_function(tokenizer, examples):
    """
    Tokenizes the input examples using the specified tokenizer.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        examples (pd.DataFrame): The examples to tokenize.

    Returns:
        dict: The tokenized examples.
    """
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128
    )


def main(data_path):
    """
    Main function to train the LLaMA chat bot model.

    Args:
        data_path (str): Path to the dataset file.
    """
    # Load and preprocess data
    data = load_data(data_path)

    # Initialize the tokenizer and model
    tokenizer = LLaMATokenizer.from_pretrained("path_to_pretrained_model")
    model = LLaMAForCausalLM.from_pretrained("path_to_pretrained_model")

    # Tokenize the data
    tokenized_data = data["text"].apply(lambda x: tokenize_function(tokenizer, x))

    # Create a data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a custom AI chat bot using Meta's LLaMA model."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset file"
    )
    args = parser.parse_args()

    main(args.data_path)
