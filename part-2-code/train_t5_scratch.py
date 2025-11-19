import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_optimizer_and_scheduler, save_model, setup_wandb
from t5_utils_scratch import initialize_model_scratch, apply_weight_init
from transformers import GenerationConfig, T5TokenizerFast
from load_data_scratch import load_t5_data_scratch
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

print(f"Using device: {DEVICE}")

def get_end_token_id(tokenizer):
    """Get the token ID for 'END'"""
    # Tokenize "END" to get its ID
    end_tokens = tokenizer.encode("END", add_special_tokens=False)
    if len(end_tokens) > 0:
        return end_tokens[0]
    return None

def mask_end_token_and_after(decoder_targets, end_token_id, pad_idx=PAD_IDX):
    """
    Mask END token and everything after it in decoder targets.
    Sets END token and subsequent tokens to PAD_IDX so they're ignored in loss.
    
    Args:
        decoder_targets: Tensor of shape (batch_size, seq_len)
        end_token_id: Token ID for END
        pad_idx: Padding index to use for masking
        
    Returns:
        Masked decoder targets
    """
    if end_token_id is None:
        return decoder_targets
    
    masked_targets = decoder_targets.clone()
    batch_size, seq_len = decoder_targets.shape
    
    for i in range(batch_size):
        # Find first occurrence of END token
        end_positions = (decoder_targets[i] == end_token_id).nonzero(as_tuple=True)[0]
        
        if len(end_positions) > 0:
            # Get position of first END token
            end_pos = end_positions[0].item()
            # Mask END and everything after it
            masked_targets[i, end_pos:] = pad_idx
    
    return masked_targets

def strip_end_token(sql_query):
    """
    Remove END token from generated SQL query.
    Handles both ' END' and 'END' at the end of the string.
    """
    sql_query = sql_query.strip()
    if sql_query.endswith(' END'):
        return sql_query[:-4].strip()
    elif sql_query.endswith('END'):
        return sql_query[:-3].strip()
    return sql_query

def get_args():
    '''
    Arguments for scratch training with optimizations
    '''
    parser = argparse.ArgumentParser(description='T5 Training from Scratch for Text-to-SQL')

    # Model hyperparameters
    parser.add_argument('--use_schema', action='store_true', default=True, help="Use schema context")
    parser.add_argument('--no_schema', dest='use_schema', action='store_false')
    
    # Scratch-specific hyperparameters
    parser.add_argument('--dropout_rate', type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing (default: 0.1)")

    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="Weight decay")
    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument('--num_warmup_epochs', type=int, default=5, help="Warmup epochs")
    parser.add_argument('--max_n_epochs', type=int, default=100, help="Max training epochs")
    parser.add_argument('--patience_epochs', type=int, default=20, help="Early stopping patience")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Gradient accumulation")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm for clipping")
    
    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=8, help="Training batch size")
    parser.add_argument('--test_batch_size', type=int, default=16, help="Eval batch size")
    parser.add_argument('--use_preprocessed', action='store_true', help="Use preprocessed data")
    parser.add_argument('--heavy_augmentation', action='store_true', help="Use heavily augmented data")
    
    # Generation hyperparameters
    parser.add_argument('--max_gen_length', type=int, default=512, help="Max generation length")
    parser.add_argument('--num_beams', type=int, default=1, help="Beam search width (1=greedy)")
    
    # Experiment tracking
    parser.add_argument('--use_wandb', action='store_true', help="Use Weights & Biases")
    parser.add_argument('--experiment_name', type=str, default='scratch_exp', help="Experiment name")
    parser.add_argument('--run_error_analysis', action='store_true', help="Run error analysis")
    parser.add_argument('--eval_every_n_epochs', type=int, default=10, help="Detailed eval frequency")
    
    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler, tokenizer):
    best_f1 = -1
    best_epoch = 0
    epochs_since_improvement = 0

    checkpoint_dir = os.path.join('checkpoints', 't5_scratch', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    
    # Setup result paths
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    
    # Get END token ID for masking in loss calculation
    end_token_id = get_end_token_id(tokenizer)
    if end_token_id is not None:
        print(f"END token ID: {end_token_id}")
        print("END token and everything after will be excluded from loss calculation")
    else:
        print("⚠ WARNING: Could not find END token ID")
    
    print("\n" + "="*80)
    print(f"SCRATCH TRAINING: {args.experiment_name}")
    print("="*80)
    print(f"Schema: {args.use_schema}")
    print(f"LR: {args.learning_rate}, WD: {args.weight_decay}")
    print(f"Scheduler: {args.scheduler_type}, Warmup: {args.num_warmup_epochs} epochs")
    print(f"Batch size: {args.batch_size}, Grad accum: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Max epochs: {args.max_n_epochs}, Patience: {args.patience_epochs}")
    print(f"Dropout: {args.dropout_rate}, Label smoothing: {args.label_smoothing}")
    print(f"Max grad norm: {args.max_grad_norm}")
    print(f"Heavy augmentation: {args.heavy_augmentation}")
    print(f"Format: Question/Answer with END tokens")
    print(f"Loss: Excludes END token and everything after")
    print("="*80 + "\n")
    
    # Initialize wandb
    use_wandb = False
    if args.use_wandb:
        use_wandb = setup_wandb_scratch(args) 
    
    for epoch in range(args.max_n_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.max_n_epochs}")
        print(f"{'='*80}")
        
        # Training
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler, end_token_id)
        print(f"Train Loss: {tr_loss:.4f}")

        # Decide evaluation frequency
        do_detailed_eval = (epoch % args.eval_every_n_epochs == 0) or (epoch == args.max_n_epochs - 1)
        
        if do_detailed_eval:
            print("Running DETAILED evaluation (with generation)...")
            eval_results = eval_epoch(args, model, dev_loader, tokenizer, epoch, end_token_id)
            
            eval_loss = eval_results['loss']
            record_f1 = eval_results['record_f1']
            record_em = eval_results['record_em']
            sql_em = eval_results['sql_em']
            error_rate = eval_results['error_rate']
            num_syntax_errors = eval_results['num_syntax_errors']
            
            print(f"Dev Loss: {eval_loss:.4f}")
            print(f"Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
            print(f"Syntax Errors: {num_syntax_errors} ({error_rate*100:.2f}%)")
            
            # Print sample predictions
            print("\n" + "-"*80)
            print("SAMPLE PREDICTIONS:")
            print("-"*80)
            for i, example in enumerate(eval_results['examples'][:3]):
                print(f"\nExample {i+1}:")
                print(f"  NL: {example['nl'][:80]}...")
                print(f"  Predicted: {example['pred'][:80]}...")
                print(f"  Gold: {example['gold'][:80]}...")
                print(f"  Match: {'✓' if example['match'] else '✗'}")
                if example['error']:
                    print(f"  ERROR: {example['error'][:60]}...")
            print("-"*80 + "\n")
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': tr_loss,
                    'dev/loss': eval_loss,
                    'dev/record_f1': record_f1,
                    'dev/record_em': record_em,
                    'dev/sql_em': sql_em,
                    'dev/error_rate': error_rate,
                    'dev/num_syntax_errors': num_syntax_errors,
                })
            
            # Check for improvement
            if record_f1 > best_f1:
                best_f1 = record_f1
                best_epoch = epoch + 1
                epochs_since_improvement = 0
                save_model(checkpoint_dir, model, best=True)
                print(f"✓ New best model! F1: {best_f1:.4f}")
            else:
                epochs_since_improvement += 1
                print(f"No improvement for {epochs_since_improvement} epoch(s)")
        
        else:
            # Quick eval - only compute loss
            print("Running QUICK evaluation (loss only)...")
            eval_loss = eval_epoch_quick(args, model, dev_loader, end_token_id)
            print(f"Dev Loss: {eval_loss:.4f}")
            
            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': tr_loss,
                    'dev/loss': eval_loss,
                })
        
        # Save last model
        save_model(checkpoint_dir, model, best=False)

        # Early stopping (only check on detailed eval)
        if do_detailed_eval and epochs_since_improvement >= args.patience_epochs:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"{'='*80}\n")
    
    if use_wandb:
        wandb.finish()

def train_epoch(args, model, train_loader, optimizer, scheduler, end_token_id):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
        label_smoothing=args.label_smoothing
    )

    progress_bar = tqdm(train_loader, desc="Training")
    optimizer.zero_grad()
    
    for batch_idx, (encoder_input, encoder_mask, decoder_input, decoder_targets, _) in enumerate(progress_bar):
        # Move to device
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # Mask END token and everything after it
        masked_targets = mask_end_token_and_after(decoder_targets, end_token_id)

        # Forward pass
        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )
        logits = outputs.logits

        # Compute loss (END token and after are now masked as PAD_IDX)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            masked_targets.reshape(-1)
        )
        
        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights after accumulation steps
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        # Track metrics
        with torch.no_grad():
            non_pad = masked_targets != PAD_IDX
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * args.gradient_accumulation_steps * num_tokens
            total_tokens += num_tokens
        
        if batch_idx % 10 == 0:
            progress_bar.set_postfix({'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}'})

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss

def eval_epoch_quick(args, model, dev_loader, end_token_id):
    """Quick evaluation - compute loss only."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
        label_smoothing=args.label_smoothing
    )
    
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader, desc="Quick Eval"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            
            # Mask END token and everything after it
            masked_targets = mask_end_token_and_after(decoder_targets, end_token_id)
            
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )
            logits = outputs.logits
            
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                masked_targets.reshape(-1)
            )
            
            non_pad = masked_targets != PAD_IDX
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss
        
def eval_epoch(args, model, dev_loader, tokenizer, epoch, end_token_id):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
        label_smoothing=args.label_smoothing
    )
    
    sql_queries = []
    
    progress_bar = tqdm(dev_loader, desc="Detailed Eval")
    
    with torch.no_grad():
        for batch_idx, (encoder_input, encoder_mask, decoder_input, decoder_targets, _) in enumerate(progress_bar):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            
            # Mask END token and everything after it
            masked_targets = mask_end_token_and_after(decoder_targets, end_token_id)
            
            # Compute loss
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )
            logits = outputs.logits
            
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                masked_targets.reshape(-1)
            )
            
            non_pad = masked_targets != PAD_IDX
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Generate SQL queries
            if args.num_beams > 1:
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_length=args.max_gen_length,
                    num_beams=args.num_beams,
                    early_stopping=True,
                )
            else:
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_length=args.max_gen_length,
                )
            
            # Decode SQL and strip END token
            for gen_ids in generated_ids:
                sql = tokenizer.decode(gen_ids, skip_special_tokens=True)
                # Remove END token from generated sequence
                sql = strip_end_token(sql)
                sql_queries.append(sql)
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    
    # Load ground truth (without END tokens)
    gt_sql_path = 'data/dev.sql'
    with open(gt_sql_path, 'r') as f:
        gt_queries = [line.strip() for line in f.readlines()]
    
    # Load NL queries
    nl_path = 'data/dev.nl'
    with open(nl_path, 'r') as f:
        nl_queries = [line.strip() for line in f.readlines()]
    
    # Save queries (without END tokens)
    model_sql_path = f'results/t5_scratch_{args.experiment_name}_dev_epoch{epoch}.sql'
    model_record_path = f'records/t5_scratch_{args.experiment_name}_dev_epoch{epoch}.pkl'
    
    with open(model_sql_path, 'w') as f:
        for sql in sql_queries:
            f.write(sql + '\n')
    
    # Compute records
    save_queries_and_records(sql_queries, model_sql_path, model_record_path)
    
    # Load ground truth records
    gt_record_path = 'records/ground_truth_dev.pkl'
    if not os.path.exists(gt_record_path):
        with open(gt_sql_path, 'r') as f:
            gt_queries_for_records = [line.strip() for line in f.readlines()]
        save_queries_and_records(gt_queries_for_records, gt_sql_path, gt_record_path)
    
    # Compute metrics
    sql_em, record_em, record_f1, _ = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )

    # Count syntax errors
    import pickle
    with open(model_record_path, 'rb') as f:
        records, error_msgs = pickle.load(f)

    num_syntax_errors = sum(1 for msg in error_msgs if msg)
    error_rate = num_syntax_errors / len(error_msgs) if error_msgs else 0

    # Prepare examples
    examples = []
    for i in range(min(10, len(sql_queries))):
        examples.append({
            'nl': nl_queries[i],
            'pred': sql_queries[i],
            'gold': gt_queries[i],
            'match': sql_queries[i].strip() == gt_queries[i].strip(),
            'error': error_msgs[i] if error_msgs[i] else None
        })

    return {
        'loss': avg_loss,
        'record_f1': record_f1,
        'record_em': record_em,
        'sql_em': sql_em,
        'error_rate': error_rate,
        'num_syntax_errors': num_syntax_errors,
        'sql_queries': sql_queries,
        'examples': examples,
    }

def test_inference(args, model, test_loader, tokenizer, model_sql_path, model_record_path):
    model.eval()
    sql_queries = []
    
    print(f"\nGenerating SQL for test set...")
    progress_bar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in progress_bar:
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate
            if args.num_beams > 1:
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_length=args.max_gen_length,
                    num_beams=args.num_beams,
                    early_stopping=True,
                )
            else:
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_length=args.max_gen_length,
                )
            
            # Decode and strip END token
            for gen_ids in generated_ids:
                sql = tokenizer.decode(gen_ids, skip_special_tokens=True)
                # Remove END token from generated sequence
                sql = strip_end_token(sql)
                sql_queries.append(sql)
    
    # Save (without END tokens)
    save_queries_and_records(sql_queries, model_sql_path, model_record_path)

    print(f"✓ Saved {len(sql_queries)} queries to {model_sql_path}")
    print(f"✓ Saved records to {model_record_path}")
    
    return sql_queries

def load_model_from_checkpoint(args, best=True):
    """Load scratch model from checkpoint"""
    checkpoint_file = 'best_model.pt' if best else 'last_model.pt'
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
    
    print(f"\nLoading model from {checkpoint_path}")
    
    model = initialize_model_scratch(args)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def setup_wandb_scratch(args):
    """Initialize Weights & Biases for scratch training."""
    try:
        import wandb
        wandb.init(
            project="t5-text-to-sql-scratch",
            name=args.experiment_name,
            config={
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler_type,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "dropout_rate": args.dropout_rate,
                "label_smoothing": args.label_smoothing,
                "use_schema": args.use_schema,
                "max_epochs": args.max_n_epochs,
                "warmup_epochs": args.num_warmup_epochs,
                "heavy_augmentation": args.heavy_augmentation,
                "eval_every_n_epochs": args.eval_every_n_epochs,
                "format": "Question/Answer with END",
            }
        )
        print("✓ Weights & Biases initialized")
        return True
    except Exception as e:
        print(f"⚠ wandb initialization failed: {e}")
        return False

def main():
    args = get_args()
    
    print("\n" + "="*80)
    print("T5 SCRATCH TRAINING for Text-to-SQL")
    print("="*80)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*80 + "\n")
    
    # Determine data folder based on augmentation type
    if args.heavy_augmentation:
        # Modify load_data to use heavy augmentation folder
        print("⚠ Using HEAVY augmentation data")
        print("⚠ Make sure you ran preprocess_data_heavy.py first!\n")
        # We'll need to modify load_data or use a custom path
    
    # Load tokenizer and data
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    train_loader, dev_loader, test_loader = load_t5_data_scratch(
        args.batch_size, args.test_batch_size,
        use_schema=args.use_schema,
        use_preprocessed=args.use_preprocessed,
        use_heavy_aug=args.heavy_augmentation
    )
    
    # Initialize model from scratch
    model = initialize_model_scratch(args)
    
    # Apply custom weight initialization
    apply_weight_init(model)
    
    # Initialize optimizer and scheduler
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
    
    # Train
    train(args, model, train_loader, dev_loader, optimizer, scheduler, tokenizer)
    
    # Load best model
    print("\nLoading best model for final evaluation...")
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Get END token ID
    end_token_id = get_end_token_id(tokenizer)
    
    # Final dev evaluation
    print("\nFinal dev set evaluation...")
    eval_results = eval_epoch(args, model, dev_loader, tokenizer, epoch=999, end_token_id=end_token_id)
    print(f"Final Dev F1: {eval_results['record_f1']:.4f}")

    # Run error analysis if requested
    if args.run_error_analysis:
        print("\nRunning error analysis...")
        pred_sql_path = f'results/t5_scratch_{args.experiment_name}_dev_epoch999.sql'
        analysis_output = f'results/error_analysis_{args.experiment_name}.txt'
        
        os.system(f'python error_analysis.py --pred_sql {pred_sql_path} --output {analysis_output}')
        print(f"✓ Error analysis saved to {analysis_output}")
        
    # Test inference
    print("\nGenerating test set predictions...")
    test_sql_path = f'results/t5_scratch_{args.experiment_name}_test.sql'
    test_record_path = f'records/t5_scratch_{args.experiment_name}_test.pkl'
    test_queries = test_inference(args, model, test_loader, tokenizer, test_sql_path, test_record_path)
    
    print("\n" + "="*80)
    print("✓ Scratch training complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()