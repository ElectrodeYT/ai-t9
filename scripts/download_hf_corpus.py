#!/usr/bin/env python3
"""Download a HuggingFace dataset and extract the 'text' column as a plain-text corpus.

Output format: UTF-8 plain text, one sentence per line.
This matches the format expected by the ai-t9 build_vocab and train scripts.

Usage
-----
    python scripts/download_hf_corpus.py --dataset HuggingFaceTB/smollm-corpus
    python scripts/download_hf_corpus.py --dataset wikitext/wikitext-103-raw-v1 --output corpuses/wikitext.txt

The script assumes the dataset has a 'text' column containing the textual data.
It splits the text into sentences using NLTK and writes each sentence on a separate line.
"""

import argparse
import os
from datasets import load_dataset
import nltk

# Enable hf_transfer for faster downloads
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

# Ensure NLTK punkt tokenizer is available
nltk.download('punkt', quiet=True)

def main():
    parser = argparse.ArgumentParser(description="Download HF dataset and extract text corpus.")
    parser.add_argument('--dataset', required=True, help='HuggingFace dataset identifier (e.g., owner/dataset)')
    parser.add_argument('--output', help='Output file path (default: corpuses/<dataset_name>.txt)')
    parser.add_argument('--config-name', help='Optional config name for the dataset (if applicable)')
    parser.add_argument('--split', help='Optional split name to use (default: all splits)')
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        dataset_name = args.dataset.split('/')[-1]
        output_path = f'corpuses/{dataset_name}.txt'

    # Ensure corpuses directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, args.config_name, split=args.split)

    # Handle single split vs multiple splits
    if args.split:
        splits = {args.split: dataset}
    else:
        splits = dataset

    # Open output file
    with open(output_path, 'w', encoding='utf-8') as f:
        # Process each split
        for split_name, split_data in splits.items():
            print(f"Processing split: {split_name}")
            for example in split_data:
                text = example['text']
                # Split into sentences
                sentences = nltk.sent_tokenize(text)
                for sentence in sentences:
                    # Strip whitespace and write
                    sentence = sentence.strip()
                    if sentence:
                        f.write(sentence + '\n')

    print(f"Corpus extracted to: {output_path}")

if __name__ == '__main__':
    main()