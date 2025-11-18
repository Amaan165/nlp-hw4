import re
import os
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Import from original preprocessing
import sys
sys.path.append('.')
from preprocess_data import (
    SIGNIFICANT_WORDS, PROTECTED_WORDS,
    get_wordnet_pos, get_synonyms, is_significant_word,
    augment_with_synonyms, preprocess_nl_query, preprocess_sql_query,
    preprocess_file_only
)

def create_multiple_augmentations(text, num_augmentations=3, aug_prob=0.6):
    """
    Create multiple augmented versions of the same text
    Each augmentation will have different random replacements
    """
    augmentations = []
    for _ in range(num_augmentations):
        aug = augment_with_synonyms(text, aug_prob=aug_prob)
        # Only add if it's actually different from original
        if aug != text:
            augmentations.append(aug)
    return augmentations

def create_heavily_augmented_dataset(
    input_nl_path, input_sql_path,
    output_nl_path, output_sql_path,
    augmentation_ratio=0.30,  # 30% augmentation
    aug_prob=0.7,  # 70% word replacement probability
    multiple_per_sample=False  # Create multiple augmentations per sample
):
    """
    Heavy augmentation pipeline for scratch training
    
    Args:
        input_nl_path: Original natural language file
        input_sql_path: Original SQL file
        output_nl_path: Output preprocessed + augmented NL file
        output_sql_path: Output preprocessed + augmented SQL file
        augmentation_ratio: Ratio of augmented examples (0.30 = 30%)
        aug_prob: Probability of replacing each significant word (0.7 = 70%)
        multiple_per_sample: If True, create 2-3 augmentations per selected sample
    """
    # Read original data
    with open(input_nl_path, 'r', encoding='utf-8') as f:
        nl_queries = [line.strip() for line in f.readlines()]
    
    with open(input_sql_path, 'r', encoding='utf-8') as f:
        sql_queries = [line.strip() for line in f.readlines()]
    
    assert len(nl_queries) == len(sql_queries), "NL and SQL files must have same length"
    
    print(f"\n{'='*80}")
    print(f"HEAVY AUGMENTATION FOR SCRATCH TRAINING")
    print(f"{'='*80}")
    print(f"Original examples: {len(nl_queries)}")
    print(f"Augmentation ratio: {augmentation_ratio*100:.1f}%")
    print(f"Word replacement probability: {aug_prob*100:.0f}%")
    print(f"Multiple augmentations per sample: {multiple_per_sample}")
    
    # Calculate number of examples to augment
    num_to_augment = int(len(nl_queries) * augmentation_ratio)
    
    # Randomly select examples to augment
    indices_to_augment = random.sample(range(len(nl_queries)), num_to_augment)
    
    # Create augmented examples (BEFORE preprocessing)
    augmented_nl_raw = []
    augmented_sql_raw = []
    
    for idx in indices_to_augment:
        if multiple_per_sample:
            # Create 2-3 different augmentations per sample
            num_versions = random.randint(2, 3)
            augmentations = create_multiple_augmentations(
                nl_queries[idx], 
                num_augmentations=num_versions,
                aug_prob=aug_prob
            )
            for aug in augmentations:
                augmented_nl_raw.append(aug)
                augmented_sql_raw.append(sql_queries[idx])
        else:
            # Single augmentation per sample
            aug_nl = augment_with_synonyms(nl_queries[idx], aug_prob=aug_prob)
            augmented_nl_raw.append(aug_nl)
            augmented_sql_raw.append(sql_queries[idx])
    
    # Combine original + augmented
    combined_nl_raw = nl_queries + augmented_nl_raw
    combined_sql_raw = sql_queries + augmented_sql_raw
    
    print(f"Augmented examples created: {len(augmented_nl_raw)}")
    print(f"Total examples before preprocessing: {len(combined_nl_raw)}")
    
    # Show augmentation examples
    print(f"\n{'='*80}")
    print(f"HEAVY AUGMENTATION EXAMPLES (showing first 5)")
    print(f"{'='*80}")
    for i in range(min(5, len(indices_to_augment))):
        orig_idx = indices_to_augment[i]
        print(f"\nExample {i+1} (original index {orig_idx}):")
        print(f"  Original:  {nl_queries[orig_idx]}")
        if multiple_per_sample:
            # Find all augmentations for this index
            count = 0
            for j, aug in enumerate(augmented_nl_raw):
                if sql_queries[orig_idx] == augmented_sql_raw[j]:
                    count += 1
                    if count <= 3:  # Show up to 3
                        print(f"  Aug #{count}:    {aug}")
        else:
            # Find the augmentation for this index
            aug_idx = indices_to_augment.index(orig_idx)
            print(f"  Augmented: {augmented_nl_raw[aug_idx]}")
    
    # Preprocess everything
    print(f"\n{'='*80}")
    print(f"PREPROCESSING")
    print(f"{'='*80}")
    
    preprocessed_nl = []
    preprocessed_sql = []
    
    for nl, sql in zip(combined_nl_raw, combined_sql_raw):
        preprocessed_nl.append(preprocess_nl_query(nl))
        preprocessed_sql.append(preprocess_sql_query(sql))
    
    # Write final output
    os.makedirs(os.path.dirname(output_nl_path) if os.path.dirname(output_nl_path) else '.', exist_ok=True)
    
    with open(output_nl_path, 'w', encoding='utf-8') as f:
        for line in preprocessed_nl:
            f.write(line + '\n')
    
    with open(output_sql_path, 'w', encoding='utf-8') as f:
        for line in preprocessed_sql:
            f.write(line + '\n')
    
    print(f"Heavy augmentation complete!")
    print(f"Final dataset size: {len(preprocessed_nl)} examples")
    print(f"Data multiplication factor: {len(preprocessed_nl) / len(nl_queries):.2f}x")
    print(f"Saved to: {output_nl_path}, {output_sql_path}")
    
    return indices_to_augment

def main():
    """
    Heavy augmentation pipeline for scratch training
    """
    print("="*80)
    print("HEAVY DATA AUGMENTATION FOR SCRATCH TRAINING")
    print("="*80)
    
    # Configuration for scratch training
    AUGMENTATION_RATIO = 0.30   # Add 30% augmented examples
    AUG_PROBABILITY = 0.7       # Replace 70% of significant words
    MULTIPLE_PER_SAMPLE = True  # Create 2-3 versions per sample
    
    print(f"\nConfiguration:")
    print(f"  Augmentation ratio: {AUGMENTATION_RATIO*100:.0f}%")
    print(f"  Word replacement probability: {AUG_PROBABILITY*100:.0f}%")
    print(f"  Multiple versions per sample: {MULTIPLE_PER_SAMPLE}")
    
    # Create output directory
    output_dir = 'data_preprocessed_heavy'
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # TRAIN SET: Heavy Augmentation + Preprocess
    # ========================================================================
    print("\n" + "="*80)
    print("PROCESSING TRAIN SET (with HEAVY augmentation)")
    print("="*80)
    
    create_heavily_augmented_dataset(
        'data/train.nl',
        'data/train.sql',
        f'{output_dir}/train.nl',
        f'{output_dir}/train.sql',
        augmentation_ratio=AUGMENTATION_RATIO,
        aug_prob=AUG_PROBABILITY,
        multiple_per_sample=MULTIPLE_PER_SAMPLE
    )
    
    # ========================================================================
    # DEV SET: Preprocess only (NO augmentation)
    # ========================================================================
    print("\n" + "="*80)
    print("PROCESSING DEV SET (no augmentation)")
    print("="*80)
    
    preprocess_file_only('data/dev.nl', f'{output_dir}/dev.nl', is_sql=False)
    preprocess_file_only('data/dev.sql', f'{output_dir}/dev.sql', is_sql=True)
    
    # ========================================================================
    # TEST SET: Preprocess only (NO augmentation, no SQL)
    # ========================================================================
    print("\n" + "="*80)
    print("PROCESSING TEST SET (no augmentation)")
    print("="*80)
    
    preprocess_file_only('data/test.nl', f'{output_dir}/test.nl', is_sql=False)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("HEAVY AUGMENTATION COMPLETE!")
    print("="*80)
    print(f"\nAll files saved to: {output_dir}/")
    print("\nExpected data size increase:")
    print(f"  Original train: 450 examples")
    if MULTIPLE_PER_SAMPLE:
        print(f"  With 30% augmentation × 2-3 versions: ~585-720 examples")
    else:
        print(f"  With 30% augmentation: ~585 examples")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Use train_t5_scratch.py with --use_preprocessed flag")
    print("2. Monitor training closely - expect slow initial progress")
    print("3. Be patient - scratch training needs 100+ epochs")

if __name__ == "__main__":
    main()