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

# ============================================================================
# PART 1: DATA AUGMENTATION
# ============================================================================

# Significant words that should be replaced with synonyms
SIGNIFICANT_WORDS = {
    # Action verbs
    'arrive', 'arrives', 'arriving',
    'leave', 'leaves', 'leaving',
    'depart', 'departs', 'departing',
    'return', 'returns', 'returning',
    'fly', 'flying',
    'show', 'list', 'give', 'find', 'get',
    'go', 'going', 'come', 'coming',
    
    # Time/sequence adjectives
    'earliest', 'latest', 'first', 'last',
    'early', 'late',
    
    # Cost adjectives
    'expensive', 'cheap', 'cheapest',
    
    # Flight type adjectives
    'nonstop', 'direct',
    
    # Important nouns
    'flight', 'flights',
    'fare', 'fares',
    'ticket', 'tickets',
    'aircraft', 'plane',
    'airport',
    'airline', 'airlines',
}

# Words to absolutely NOT replace
PROTECTED_WORDS = {
    # Query structure words
    'what', 'which', 'where', 'when', 'how', 'who',
    'all', 'any', 'some',
    
    # Prepositions (important for SQL structure)
    'from', 'to', 'in', 'on', 'at', 'by', 'with', 'and', 'or', 'between', 'before', 'after',
    
    # Time/date terms
    'morning', 'afternoon', 'evening', 'night', 'day', 'week', 'month', 'year',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',
    
    # Service class
    'class', 'economy', 'business',
    'meal', 'meals',
    'stop', 'stops', 'stopover',
}

def get_wordnet_pos(treebank_tag):
    """Convert Penn Treebank POS tag to WordNet POS tag"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def get_synonyms(word, pos=None):
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym.lower())
    
    return list(synonyms)

def is_significant_word(token, pos_tag):
    """Determine if a word is significant enough to be replaced"""
    token_lower = token.lower()
    
    # Protected words should never be replaced
    if token_lower in PROTECTED_WORDS:
        return False
    
    # Explicitly significant words should always be replaced
    if token_lower in SIGNIFICANT_WORDS:
        return True
    
    # Also consider verbs (VB*) and adjectives (JJ*) as significant
    if pos_tag.startswith('VB') or pos_tag.startswith('JJ'):
        return True
    
    return False

def augment_with_synonyms(text, aug_prob=0.5):
    """Augment text by replacing SIGNIFICANT words with synonyms"""
    # Tokenize and POS tag
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    augmented_tokens = []
    
    for token, pos in pos_tags:
        # Skip if not alphabetic
        if not token.isalpha():
            augmented_tokens.append(token)
            continue
        
        # Check if this is a significant word
        if not is_significant_word(token, pos):
            augmented_tokens.append(token)
            continue
        
        # Random chance to skip (even for significant words)
        if random.random() > aug_prob:
            augmented_tokens.append(token)
            continue
        
        # Get WordNet POS
        wordnet_pos = get_wordnet_pos(pos)
        if wordnet_pos is None:
            augmented_tokens.append(token)
            continue
        
        # Get synonyms
        synonyms = get_synonyms(token.lower(), pos=wordnet_pos)
        
        if synonyms:
            # Randomly select one synonym
            synonym = random.choice(synonyms)
            # Preserve original capitalization
            if token[0].isupper():
                synonym = synonym.capitalize()
            augmented_tokens.append(synonym)
        else:
            augmented_tokens.append(token)
    
    return ' '.join(augmented_tokens)

# ============================================================================
# PART 2: PREPROCESSING
# ============================================================================

def preprocess_nl_query(query):
    """
    Preprocess natural language query
    
    Preprocessing steps:
    1. Lowercase
    2. Normalize query phrases (give me, show me, what flights, etc.) -> list
    3. Remove extra whitespace
    4. Remove punctuation at the end
    """
    # Step 1: Lowercase
    query = query.lower()
    
    # Step 2: Normalize query phrases to "list"
    # Handle various question patterns
    query = re.sub(r'^give me (the )?', 'list ', query)
    query = re.sub(r'^show me (the )?', 'list ', query)
    query = re.sub(r'^provide me (the )?', 'list ', query)
    query = re.sub(r'^tell me (the )?', 'list ', query)
    query = re.sub(r'^find me (the )?', 'list ', query)
    query = re.sub(r'^get me (the )?', 'list ', query)
    query = re.sub(r'^i want (the )?', 'list ', query)
    query = re.sub(r'^i need (the )?', 'list ', query)
    query = re.sub(r'^i would like (the )?', 'list ', query)
    query = re.sub(r"^i'd like (the )?", 'list ', query)
    
    # Handle "what" questions
    query = re.sub(r'^what (is |are )?(the )?', 'list ', query)
    query = re.sub(r'^which (is |are )?(the )?', 'list ', query)
    
    # Handle "can you" questions
    query = re.sub(r'^can you ', 'list ', query)
    query = re.sub(r'^could you ', 'list ', query)
    
    # Step 3: Remove extra whitespace
    query = re.sub(r'\s+', ' ', query)
    
    # Step 4: Remove trailing punctuation
    query = query.rstrip('?.,;:!')
    
    # Step 5: Strip leading/trailing whitespace
    query = query.strip()
    
    return query

def preprocess_sql_query(query):
    """Minimal preprocessing for SQL query - just normalize whitespace"""
    query = re.sub(r'\s+', ' ', query)
    query = query.strip()
    return query

# ============================================================================
# PART 3: COMBINED PIPELINE
# ============================================================================

def create_augmented_and_preprocessed_dataset(
    input_nl_path, input_sql_path, 
    output_nl_path, output_sql_path, 
    augmentation_ratio=0.05, aug_prob=0.5):
    """
    Complete pipeline: Augment + Preprocess data
    
    Args:
        input_nl_path: Original natural language file
        input_sql_path: Original SQL file
        output_nl_path: Output preprocessed + augmented NL file
        output_sql_path: Output preprocessed + augmented SQL file
        augmentation_ratio: Ratio of augmented examples to add (e.g., 0.05 = 5%)
        aug_prob: Probability of replacing each significant word
    """
    # Read original data
    with open(input_nl_path, 'r', encoding='utf-8') as f:
        nl_queries = [line.strip() for line in f.readlines()]
    
    with open(input_sql_path, 'r', encoding='utf-8') as f:
        sql_queries = [line.strip() for line in f.readlines()]
    
    assert len(nl_queries) == len(sql_queries), "NL and SQL files must have same length"
    
    # Calculate number of examples to augment
    num_to_augment = int(len(nl_queries) * augmentation_ratio)
    
    print(f"\n{'='*80}")
    print(f"STEP 1: AUGMENTATION")
    print(f"{'='*80}")
    print(f"Original examples: {len(nl_queries)}")
    print(f"Augmented examples to add: {num_to_augment}")
    print(f"Augmentation ratio: {augmentation_ratio*100:.1f}%")
    
    # Randomly select examples to augment
    indices_to_augment = random.sample(range(len(nl_queries)), num_to_augment)
    
    # Create augmented examples (BEFORE preprocessing)
    augmented_nl_raw = []
    augmented_sql_raw = []
    
    for idx in indices_to_augment:
        aug_nl = augment_with_synonyms(nl_queries[idx], aug_prob=aug_prob)
        augmented_nl_raw.append(aug_nl)
        augmented_sql_raw.append(sql_queries[idx])
    
    # Combine original + augmented
    combined_nl_raw = nl_queries + augmented_nl_raw
    combined_sql_raw = sql_queries + augmented_sql_raw
    
    print(f"Total examples before preprocessing: {len(combined_nl_raw)}")
    
    # Show augmentation examples
    print(f"\n{'='*80}")
    print(f"AUGMENTATION EXAMPLES (showing first 5)")
    print(f"{'='*80}")
    for i in range(min(5, len(indices_to_augment))):
        orig_idx = indices_to_augment[i]
        print(f"\nExample {i+1} (original index {orig_idx}):")
        print(f"  Original:  {nl_queries[orig_idx]}")
        print(f"  Augmented: {augmented_nl_raw[i]}")
    
    # Now preprocess everything
    print(f"\n{'='*80}")
    print(f"STEP 2: PREPROCESSING")
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
    
    print(f"Preprocessing complete!")
    print(f"Final dataset size: {len(preprocessed_nl)} examples")
    print(f"Saved to: {output_nl_path}, {output_sql_path}")
    
    # Show preprocessing examples
    print(f"\n{'='*80}")
    print(f"PREPROCESSING EXAMPLES (showing first 5)")
    print(f"{'='*80}")
    for i in range(min(5, len(nl_queries))):
        print(f"\nExample {i+1}:")
        print(f"  Original:     {nl_queries[i]}")
        print(f"  Preprocessed: {preprocessed_nl[i]}")
    
    return indices_to_augment

def preprocess_file_only(input_path, output_path, is_sql=False):
    """
    Preprocess a file line by line (NO augmentation, just preprocessing)
    Used for dev and test sets
    """
    with open(input_path, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    preprocessed_lines = []
    for line in lines:
        line = line.strip()
        if line:
            if is_sql:
                preprocessed_line = preprocess_sql_query(line)
            else:
                preprocessed_line = preprocess_nl_query(line)
            preprocessed_lines.append(preprocessed_line)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for line in preprocessed_lines:
            f_out.write(line + '\n')
    
    print(f"Preprocessed {input_path} -> {output_path} ({len(preprocessed_lines)} examples)")
    return len(preprocessed_lines)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Complete data preparation pipeline
    """
    print("="*80)
    print("DATA PREPARATION: AUGMENTATION + PREPROCESSING")
    print("="*80)
    
    # Configuration
    AUGMENTATION_RATIO = 0.05  # Add 5% augmented examples (2-5% recommended)
    AUG_PROBABILITY = 0.5      # Probability of replacing each significant word (50%)
    
    print(f"\nConfiguration:")
    print(f"  Augmentation ratio: {AUGMENTATION_RATIO*100:.1f}% of training data")
    print(f"  Significant word replacement probability: {AUG_PROBABILITY}")
    print(f"\nSignificant words include:")
    print(f"  - Action verbs: arrive, leave, depart, return, show, list, etc.")
    print(f"  - Important adjectives: earliest, latest, expensive, cheap, nonstop, etc.")
    print(f"  - Key nouns: flight, fare, ticket, airline, etc.")
    
    # Create output directory
    output_dir = 'data_preprocessed'
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # TRAIN SET: Augment + Preprocess
    # ========================================================================
    print("\n" + "="*80)
    print("PROCESSING TRAIN SET (with augmentation)")
    print("="*80)
    
    create_augmented_and_preprocessed_dataset(
        'data/train.nl', 
        'data/train.sql',
        f'{output_dir}/train.nl',
        f'{output_dir}/train.sql',
        augmentation_ratio=AUGMENTATION_RATIO,
        aug_prob=AUG_PROBABILITY
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
    print("DATA PREPARATION COMPLETE!")
    print("="*80)
    print(f"\nAll preprocessed files saved to: {output_dir}/")
    print("\nFiles created:")
    print(f"  - {output_dir}/train.nl    (original + augmented + preprocessed)")
    print(f"  - {output_dir}/train.sql   (original + augmented + preprocessed)")
    print(f"  - {output_dir}/dev.nl      (preprocessed only)")
    print(f"  - {output_dir}/dev.sql     (preprocessed only)")
    print(f"  - {output_dir}/test.nl     (preprocessed only)")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Run compute_statistics.py to generate statistics")
    print("2. Use the preprocessed data for training your model")
    print("3. Report both augmentation and preprocessing details in your writeup")
    
    print("\nNote:")
    print("  - Only TRAIN set is augmented (dev/test are NOT augmented)")
    print("  - Only SIGNIFICANT words are replaced with synonyms")
    print("  - All sets are preprocessed (lowercased, normalized, etc.)")

if __name__ == "__main__":
    main()