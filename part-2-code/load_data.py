import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.use_schema = use_schema
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # Load schema if needed
        if self.use_schema:
            self.schema = self.load_schema(os.path.join(data_folder, 'flight_database.schema'))
            print(f"Schema loaded ({len(self.schema)} chars)")
        else:
            self.schema = ""
            print("Training WITHOUT schema context")
        
        # Determine data path
        if use_preprocessed and os.path.exists(os.path.join(data_folder, 'data_preprocessed')):
            data_path = os.path.join(data_folder, 'data_preprocessed')
            print(f"Using preprocessed data from {data_path}")
        else:
            data_path = data_folder
            print(f"Using original data from {data_path}")
        
        # Load data
        self.nl_queries, self.sql_queries = self.load_data(data_path, split)
        print(f"Loaded {len(self.nl_queries)} examples for {split} split")

    def load_schema(self, schema_path):
        """Load and format database schema."""
        with open(schema_path, 'r') as f:
            schema_lines = f.readlines()
        
        # Extract table and column information
        schema_text = []
        for line in schema_lines:
            line = line.strip()
            if line and not line.startswith('--'):
                # Keep CREATE TABLE and column definitions
                if 'CREATE TABLE' in line or '(' in line or ')' in line:
                    schema_text.append(line)
        
        schema_str = ' '.join(schema_text)
        # Limit schema length
        if len(schema_str) > 800:
            schema_str = schema_str[:800]
        
        return schema_str
    
    def load_data(self, data_path, split):
        """Load natural language and SQL queries."""
        nl_file = os.path.join(data_path, f'{split}.nl')
        sql_file = os.path.join(data_path, f'{split}.sql')
        
        with open(nl_file, 'r') as f:
            nl_queries = [line.strip() for line in f.readlines()]
        
        if split == 'test':
            sql_queries = [None] * len(nl_queries)
        else:
            with open(sql_file, 'r') as f:
                sql_queries = [line.strip() for line in f.readlines()]
        
        return nl_queries, sql_queries

    def process_data(self, data_folder, split, tokenizer):
        # Not implemented - data processing handled in __getitem__
        pass
    
    def __len__(self):
        return len(self.nl_queries)

    def __getitem__(self, idx):
        nl_query = self.nl_queries[idx]
        sql_query = self.sql_queries[idx]
        
        # Format input with optional schema context
        if self.use_schema:
            # Include schema for better SQL generation
            encoder_input = f"translate to SQL: {self.schema} | query: {nl_query}"
        else:
            encoder_input = f"translate to SQL: {nl_query}"
        
        # Tokenize encoder input
        encoder_tokens = self.tokenizer(
            encoder_input,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        
        encoder_ids = encoder_tokens['input_ids'].squeeze(0)
        encoder_mask = encoder_tokens['attention_mask'].squeeze(0)
        
        if self.split == 'test':
            # For test set, no target SQL
            return encoder_ids, encoder_mask, None, None, None
        else:
            # Tokenize decoder output (SQL)
            decoder_tokens = self.tokenizer(
                sql_query,
                max_length=512,
                truncation=True,
                return_tensors='pt'
            )
            
            decoder_ids = decoder_tokens['input_ids'].squeeze(0)
            
            # Prepare decoder inputs and targets
            decoder_input_ids = decoder_ids[:-1]
            decoder_target_ids = decoder_ids[1:]
            
            # Initial decoder token
            initial_decoder_input = torch.tensor([self.tokenizer.pad_token_id])
            
            return encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_input


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = []
    encoder_mask_list = []
    decoder_inputs_list = []
    decoder_targets_list = []
    initial_decoder_inputs_list = []
    
    for item in batch:
        enc_ids, enc_mask, dec_in, dec_tgt, initial_dec = item
        
        encoder_ids_list.append(enc_ids)
        encoder_mask_list.append(enc_mask)
        if dec_in is not None:
            decoder_inputs_list.append(dec_in)
            decoder_targets_list.append(dec_tgt)
            initial_decoder_inputs_list.append(initial_dec)
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask_list, batch_first=True, padding_value=0)
    
    if decoder_inputs_list:
        decoder_inputs = pad_sequence(decoder_inputs_list, batch_first=True, padding_value=PAD_IDX)
        decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)
        initial_decoder_inputs = torch.stack(initial_decoder_inputs_list)
    else:
        decoder_inputs = None
        decoder_targets = None
        initial_decoder_inputs = None
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = []
    encoder_mask_list = []
    
    for item in batch:
        enc_ids, enc_mask, _, _, _ = item
        encoder_ids_list.append(enc_ids)
        encoder_mask_list.append(enc_mask)
    
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask_list, batch_first=True, padding_value=0)
    
    batch_size = len(batch)
    initial_decoder_inputs = torch.full((batch_size, 1), PAD_IDX, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dataset = T5Dataset(data_folder, split, use_schema=use_schema, use_preprocessed=use_preprocessed)
    shuffle = (split == "train")
    collate = normal_collate_fn if split != "test" else test_collate_fn
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate,
        num_workers=0
    )
    
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    print("\n" + "="*80)
    print("Loading T5 data...")
    print(f"Schema context: {use_schema}")
    print(f"Preprocessed data: {use_preprocessed}")
    print("="*80)
    
    train_loader = get_dataloader(batch_size, "train", use_schema, use_preprocessed)
    dev_loader = get_dataloader(test_batch_size, "dev", use_schema, use_preprocessed)
    test_loader = get_dataloader(test_batch_size, "test", use_schema, use_preprocessed)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("="*80 + "\n")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x