import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    words = word_tokenize(text)
    transformed_words = []
    
    # QWERTY keyboard neighbors for typo simulation
    keyboard_neighbors = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx',
        'e': 'wrdsf', 'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb',
        'i': 'ujklo', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop',
        'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
        'q': 'wa', 'r': 'etdf', 's': 'awedxz', 't': 'ryfg',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
        'y': 'tghu', 'z': 'asx'
    }
    
    for word in words:
        # Apply transformations with probability
        if random.random() < 0.4:  # 40% chance to transform word
            choice = random.choice(['synonym', 'typo'])
            
            if choice == 'synonym':
                # Synonym replacement using WordNet
                synsets = wordnet.synsets(word)
                if synsets:
                    synonyms = []
                    for syn in synsets[:3]:  # Check first 3 synsets
                        for lemma in syn.lemmas():
                            if lemma.name() != word and '_' not in lemma.name():
                                synonyms.append(lemma.name())
                    if synonyms:
                        word = random.choice(synonyms)
            
            elif choice == 'typo' and len(word) > 2:
                # Random typo by replacing a character
                word_list = list(word.lower())
                
                # Apply 1-2 typos to the word
                num_typos = min(random.randint(1, 2), len(word_list))
                for _ in range(num_typos):
                    idx = random.randint(0, len(word_list) - 1)
                    if word_list[idx] in keyboard_neighbors:
                        word_list[idx] = random.choice(keyboard_neighbors[word_list[idx]])
                
                word = ''.join(word_list)
        
        transformed_words.append(word)
    
    # Detokenize
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_words)

    ##### YOUR CODE ENDS HERE ######

    return example
