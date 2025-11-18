import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def initialize_model_scratch(args):
    """
    Initialize T5 model from scratch with custom configuration
    optimized for small dataset training
    """
    print(f"\nInitializing T5-small from scratch with optimizations...")
    
    # Load base config
    config = T5Config.from_pretrained('google-t5/t5-small')
    
    # Adjust dropout for scratch training
    config.dropout_rate = args.dropout_rate
    config.layer_norm_epsilon = 1e-6
    
    print(f"Configuration adjustments:")
    print(f"  Dropout rate: {config.dropout_rate}")
    print(f"  Layer norm epsilon: {config.layer_norm_epsilon}")
    
    # Initialize model
    model = T5ForConditionalGeneration(config)
    model = model.to(DEVICE)
    
    # Print parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.1f} MB (FP32)")
    
    return model

def apply_weight_init(model, init_std=0.02):
    """
    Apply custom weight initialization for better scratch training
    
    Uses Xavier/Glorot initialization for most layers with small std
    to prevent gradient explosion in early training
    """
    print(f"\nApplying custom weight initialization (std={init_std})...")
    
    def _init_weights(module):
        """Initialize weights for different layer types"""
        if isinstance(module, nn.Linear):
            # Xavier initialization with small std
            module.weight.data.normal_(mean=0.0, std=init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Embedding layers
            module.weight.data.normal_(mean=0.0, std=init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # Layer norm: bias=0, weight=1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    # Apply initialization
    model.apply(_init_weights)
    
    # Special initialization for specific T5 components
    if hasattr(model, 'shared'):
        # Shared embedding layer
        model.shared.weight.data.normal_(mean=0.0, std=init_std)
    
    print("✓ Weight initialization complete")
    
    return model

def get_trainable_params_by_layer(model):
    """
    Analyze trainable parameters by layer type
    Useful for debugging and understanding the model
    """
    param_dict = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_type = name.split('.')[0]
            if layer_type not in param_dict:
                param_dict[layer_type] = 0
            param_dict[layer_type] += param.numel()
    
    print("\nTrainable parameters by layer type:")
    total = 0
    for layer_type, count in sorted(param_dict.items()):
        print(f"  {layer_type}: {count:,} params")
        total += count
    print(f"  TOTAL: {total:,} params")
    
    return param_dict

def print_model_structure(model, max_depth=2):
    """
    Print model structure for verification
    """
    print("\nModel Structure:")
    print("="*80)
    
    def print_module(module, indent=0, current_depth=0):
        if current_depth >= max_depth:
            return
        
        for name, child in module.named_children():
            print("  " * indent + f"├─ {name}: {child.__class__.__name__}")
            print_module(child, indent + 1, current_depth + 1)
    
    print_module(model)
    print("="*80)

def freeze_embeddings(model):
    """
    Optionally freeze embedding layers
    Can help with stability in early training
    """
    print("\nFreezing embedding layers...")
    
    frozen_params = 0
    if hasattr(model, 'shared'):
        for param in model.shared.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"✓ Froze {frozen_params:,} embedding parameters")
    
    return model

def unfreeze_embeddings(model):
    """
    Unfreeze embeddings for later fine-tuning
    """
    print("\nUnfreezing embedding layers...")
    
    if hasattr(model, 'shared'):
        for param in model.shared.parameters():
            param.requires_grad = True
    
    print("✓ Embeddings unfrozen")
    
    return model

def check_gradient_flow(model):
    """
    Check if gradients are flowing properly through the model
    Call this after a backward pass during training
    """
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())
    
    if len(ave_grads) == 0:
        print("⚠ No gradients found! Check if backward() was called.")
        return
    
    print("\nGradient Flow Check:")
    print(f"  Average gradient: {sum(ave_grads) / len(ave_grads):.6f}")
    print(f"  Max gradient: {max(max_grads):.6f}")
    print(f"  Layers with gradients: {len(ave_grads)}")
    
    # Check for gradient issues
    if max(max_grads) > 100:
        print("  ⚠ WARNING: Very large gradients detected! May cause instability.")
    if sum(ave_grads) / len(ave_grads) < 1e-6:
        print("  ⚠ WARNING: Very small gradients detected! May indicate vanishing gradients.")
    
    return ave_grads, max_grads, layers

def get_model_stats(model):
    """
    Get comprehensive model statistics
    """
    stats = {
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'frozen_params': sum(p.numel() for p in model.parameters() if not p.requires_grad),
        'encoder_params': sum(p.numel() for p in model.encoder.parameters()),
        'decoder_params': sum(p.numel() for p in model.decoder.parameters()),
    }
    
    stats['trainable_ratio'] = stats['trainable_params'] / stats['total_params']
    
    print("\nModel Statistics:")
    print("="*80)
    print(f"Total parameters:     {stats['total_params']:,}")
    print(f"Trainable parameters: {stats['trainable_params']:,}")
    print(f"Frozen parameters:    {stats['frozen_params']:,}")
    print(f"Trainable ratio:      {stats['trainable_ratio']*100:.2f}%")
    print(f"\nBy component:")
    print(f"  Encoder: {stats['encoder_params']:,}")
    print(f"  Decoder: {stats['decoder_params']:,}")
    print("="*80)
    
    return stats