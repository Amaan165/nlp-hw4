#!/usr/bin/env python3
"""
Generate test predictions using the best checkpoint identified from dev evaluation
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import os

class SQLDataset(Dataset):
    """Dataset for text-to-SQL generation"""
    def __init__(self, nl_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read natural language queries
        with open(nl_file, 'r') as f:
            self.queries = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        
        # Tokenize input
        encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'query': query
        }

def generate_predictions(model, dataloader, tokenizer, device, output_file):
    """Generate SQL predictions for the test set"""
    model.eval()
    predictions = []
    
    print(f"Generating predictions...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode predictions
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)
    
    # Save predictions
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(pred + '\n')
    
    print(f"\nPredictions saved to: {output_file}")
    print(f"Total predictions: {len(predictions)}")

def main():
    parser = argparse.ArgumentParser(description='Generate test predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (e.g., checkpoints/t5_ft_higherlr_cosine_batch32_40ep_epoch35.pkl)')
    parser.add_argument('--test_file', type=str, default='data/test.nl',
                       help='Path to test natural language queries')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file for predictions (e.g., results/final_test_predictions.sql)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for generation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GENERATING TEST PREDICTIONS")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test file: {args.test_file}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")
    
    # Check if files exist
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    if not os.path.exists(args.test_file):
        print(f"Error: Test file not found at {args.test_file}")
        return
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
    
    # Load model
    print("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    print(f"Model loaded on {args.device}\n")
    
    # Create dataset and dataloader
    print("Loading test data...")
    test_dataset = SQLDataset(args.test_file, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Generate predictions
    generate_predictions(model, test_loader, tokenizer, args.device, args.output)
    
    print("\n" + "="*80)
    print("DONE! Next steps:")
    print("1. Check the predictions file")
    print("2. Submit to the evaluation system")
    print("="*80)

if __name__ == "__main__":
    main()