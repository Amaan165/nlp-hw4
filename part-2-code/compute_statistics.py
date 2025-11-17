from transformers import T5TokenizerFast
from collections import Counter
import numpy as np
import os

def load_lines(path):
    """Load lines from a file"""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def compute_statistics(nl_path, sql_path=None, tokenizer=None):
    """
    Compute statistics for the dataset
    
    Args:
        nl_path: Path to natural language file (.nl)
        sql_path: Path to SQL file (.sql), optional for test set
        tokenizer: T5 tokenizer for tokenization
    """
    # Load data
    nl_queries = load_lines(nl_path)
    sql_queries = load_lines(sql_path) if sql_path else None
    
    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Number of examples
    num_examples = len(nl_queries)
    
    # Tokenize natural language queries
    nl_tokenized = [tokenizer.encode(query, add_special_tokens=False) for query in nl_queries]
    nl_lengths = [len(tokens) for tokens in nl_tokenized]
    mean_nl_length = np.mean(nl_lengths)
    
    # Vocabulary for natural language
    nl_vocab = set()
    for tokens in nl_tokenized:
        nl_vocab.update(tokens)
    nl_vocab_size = len(nl_vocab)
    
    # SQL statistics (if available)
    if sql_queries:
        sql_tokenized = [tokenizer.encode(query, add_special_tokens=False) for query in sql_queries]
        sql_lengths = [len(tokens) for tokens in sql_tokenized]
        mean_sql_length = np.mean(sql_lengths)
        
        # Vocabulary for SQL
        sql_vocab = set()
        for tokens in sql_tokenized:
            sql_vocab.update(tokens)
        sql_vocab_size = len(sql_vocab)
    else:
        mean_sql_length = None
        sql_vocab_size = None
    
    return {
        'num_examples': num_examples,
        'mean_nl_length': mean_nl_length,
        'mean_sql_length': mean_sql_length,
        'nl_vocab_size': nl_vocab_size,
        'sql_vocab_size': sql_vocab_size,
        'nl_tokenized': nl_tokenized,
        'sql_tokenized': sql_tokenized if sql_queries else None
    }

def print_statistics_table(stats, split_name):
    """Print statistics in a nice table format"""
    print(f"\n{split_name.upper()} SET:")
    print(f"  Number of examples: {stats['num_examples']}")
    print(f"  Mean sentence length: {stats['mean_nl_length']:.2f} tokens")
    if stats['mean_sql_length'] is not None:
        print(f"  Mean SQL query length: {stats['mean_sql_length']:.2f} tokens")
    else:
        print(f"  Mean SQL query length: N/A (no ground truth)")
    print(f"  Vocabulary size (natural language): {stats['nl_vocab_size']}")
    if stats['sql_vocab_size'] is not None:
        print(f"  Vocabulary size (SQL): {stats['sql_vocab_size']}")
    else:
        print(f"  Vocabulary size (SQL): N/A (no ground truth)")

def main():
    # Initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    print("="*80)
    print("TABLE 1: Data Statistics BEFORE Pre-processing")
    print("="*80)
    print("\nUsing T5 Tokenizer: google-t5/t5-small\n")
    
    # Training set - ORIGINAL
    train_stats_orig = compute_statistics('data/train.nl', 'data/train.sql', tokenizer)
    print_statistics_table(train_stats_orig, "train")
    
    # Development set - ORIGINAL
    dev_stats_orig = compute_statistics('data/dev.nl', 'data/dev.sql', tokenizer)
    print_statistics_table(dev_stats_orig, "dev")
    
    print("\n" + "="*80)
    print("TABLE 2: Data Statistics AFTER Pre-processing")
    print("="*80)
    print("\nModel name: google-t5/t5-small")
    print("Preprocessing: Lowercasing, phrase normalization, whitespace normalization\n")
    
    # Check if preprocessed data exists
    if os.path.exists('data_preprocessed/train.nl'):
        # Training set - PREPROCESSED
        train_stats_prep = compute_statistics('data_preprocessed/train.nl', 'data_preprocessed/train.sql', tokenizer)
        print_statistics_table(train_stats_prep, "train")
        
        # Development set - PREPROCESSED
        dev_stats_prep = compute_statistics('data_preprocessed/dev.nl', 'data_preprocessed/dev.sql', tokenizer)
        print_statistics_table(dev_stats_prep, "dev")
        
        # Show changes
        print("\n" + "="*80)
        print("IMPACT OF PREPROCESSING")
        print("="*80)
        print(f"\nTRAIN SET:")
        print(f"  Mean NL length change: {train_stats_orig['mean_nl_length']:.2f} -> {train_stats_prep['mean_nl_length']:.2f} ({train_stats_prep['mean_nl_length'] - train_stats_orig['mean_nl_length']:.2f})")
        print(f"  NL vocab size change: {train_stats_orig['nl_vocab_size']} -> {train_stats_prep['nl_vocab_size']} ({train_stats_prep['nl_vocab_size'] - train_stats_orig['nl_vocab_size']})")
        
        print(f"\nDEV SET:")
        print(f"  Mean NL length change: {dev_stats_orig['mean_nl_length']:.2f} -> {dev_stats_prep['mean_nl_length']:.2f} ({dev_stats_prep['mean_nl_length'] - dev_stats_orig['mean_nl_length']:.2f})")
        print(f"  NL vocab size change: {dev_stats_orig['nl_vocab_size']} -> {dev_stats_prep['nl_vocab_size']} ({dev_stats_prep['nl_vocab_size'] - dev_stats_orig['nl_vocab_size']})")
    else:
        print("\nPreprocessed data not found. Run preprocess_data.py first!")
    
    # Sample examples
    print("\n" + "="*80)
    print("SAMPLE DATA EXAMPLES (Original)")
    print("="*80)
    nl_queries = load_lines('data/train.nl')
    sql_queries = load_lines('data/train.sql')
    
    for i in range(min(3, len(nl_queries))):
        print(f"\nExample {i+1}:")
        print(f"  Natural Language: {nl_queries[i]}")
        print(f"  SQL Query: {sql_queries[i][:100]}...")  # Truncate long SQL
        print(f"  NL tokens: {len(train_stats_orig['nl_tokenized'][i])}")
        print(f"  SQL tokens: {len(train_stats_orig['sql_tokenized'][i])}")

if __name__ == "__main__":
    main()