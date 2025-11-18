import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    """Initialize Weights & Biases logging."""
    try:
        import wandb
        wandb.init(
            project="t5-text-to-sql",
            name=args.experiment_name,
            config={
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler_type,
                "batch_size": args.batch_size,
                "finetune": getattr(args, 'finetune', False),
                "use_schema": args.use_schema,
                "max_epochs": args.max_n_epochs,
                "warmup_epochs": args.num_warmup_epochs,
            }
        )
        print("✓ Weights & Biases initialized")
        return True
    except ImportError:
        print("⚠ wandb not installed. Run: pip install wandb")
        return False
    except Exception as e:
        print(f"⚠ wandb initialization failed: {e}")
        return False

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    print(f"\nInitializing T5 model (finetune={args.finetune})...")
    
    if args.finetune:
        # Fine-tune pretrained T5-small
        print("Loading pretrained google-t5/t5-small...")
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
        
        # Optional: Freeze encoder layers for faster training (uncomment if needed)
        # print("Freezing encoder layers...")
        # for param in model.encoder.parameters():
        #     param.requires_grad = False
        
    else:
        # Train from scratch
        print("Initializing T5-small from scratch...")
        config = T5Config.from_pretrained('google-t5/t5-small')
        model = T5ForConditionalGeneration(config)
    
    model = model.to(DEVICE)
    
    # Print parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable: {100 * trainable_params / total_params:.2f}%\n")
    
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    mkdir(checkpoint_dir)
    
    if best:
        save_path = os.path.join(checkpoint_dir, 'best_model.pt')
        print(f"✓ Saving best model to {save_path}")
    else:
        save_path = os.path.join(checkpoint_dir, 'last_model.pt')
    
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    checkpoint_file = 'best_model.pt' if best else 'last_model.pt'
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
    
    print(f"\nLoading model from {checkpoint_path}")
    
    model = initialize_model(args)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    # Separate parameters for weight decay
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() 
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() 
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    print(f"Optimizer: AdamW (lr={args.learning_rate}, wd={args.weight_decay})")
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, 
        eps=1e-8, 
        betas=(0.9, 0.999)
    )
    
    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    if args.scheduler_type == "none":
        print("Scheduler: None")
        return None
    
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs
    
    print(f"Scheduler: {args.scheduler_type}")
    print(f"Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
    
    if args.scheduler_type == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif args.scheduler_type == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler_type} not implemented")
    
    return scheduler

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

