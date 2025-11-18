"""
Modified data loader to support heavy augmentation for scratch training
This extends the original load_data.py with support for multiple data directories
"""

import os
from load_data import T5Dataset, normal_collate_fn, test_collate_fn
from torch.utils.data import DataLoader

def get_dataloader_scratch(batch_size, split, use_schema=True, use_preprocessed=False, use_heavy_aug=False):
    """
    Get dataloader with support for heavy augmentation
    
    Args:
        batch_size: Batch size
        split: 'train', 'dev', or 'test'
        use_schema: Whether to use schema in input
        use_preprocessed: Whether to use preprocessed data
        use_heavy_aug: Whether to use heavily augmented data (for scratch training)
    """
    data_folder = 'data'
    
    # Determine which preprocessed folder to use
    if use_heavy_aug and use_preprocessed:
        # Use heavily augmented data
        preprocessed_folder = 'data_preprocessed_heavy'
        if os.path.exists(preprocessed_folder):
            print(f"Using HEAVY augmentation data from {preprocessed_folder}")
        else:
            print(f"⚠ Heavy augmentation folder not found: {preprocessed_folder}")
            print(f"⚠ Run: python preprocess_data_heavy.py")
            print(f"⚠ Falling back to regular preprocessed data")
            preprocessed_folder = 'data_preprocessed'
    else:
        preprocessed_folder = 'data_preprocessed' if use_preprocessed else None
    
    # Create dataset with appropriate folder
    class ModifiedT5Dataset(T5Dataset):
        def __init__(self, data_folder, split, use_schema=True, preprocessed_folder=None):
            self.split = split
            self.use_schema = use_schema
            self.tokenizer = T5Dataset.__dict__['__init__'].__code__.co_consts[1]  # Get tokenizer
            
            # Import here to avoid circular dependency
            from transformers import T5TokenizerFast
            self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
            
            # Load schema if needed
            if self.use_schema:
                self.schema = self.load_schema(os.path.join(data_folder, 'flight_database.schema'))
                print(f"Schema loaded ({len(self.schema)} chars)")
            else:
                self.schema = ""
                print("Training WITHOUT schema context")
            
            # Determine data path
            if preprocessed_folder and os.path.exists(preprocessed_folder):
                data_path = preprocessed_folder
                print(f"Using preprocessed data from {data_path}")
            else:
                data_path = data_folder
                print(f"Using original data from {data_path}")
            
            # Load data
            self.nl_queries, self.sql_queries = self.load_data(data_path, split)
            print(f"Loaded {len(self.nl_queries)} examples for {split} split")
    
    dataset = ModifiedT5Dataset(
        data_folder, 
        split, 
        use_schema=use_schema,
        preprocessed_folder=preprocessed_folder
    )
    
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

def load_t5_data_scratch(batch_size, test_batch_size, use_schema=True, use_preprocessed=False, use_heavy_aug=False):
    """
    Load data for scratch training with optional heavy augmentation
    
    Args:
        batch_size: Training batch size
        test_batch_size: Eval batch size
        use_schema: Whether to use schema
        use_preprocessed: Whether to use preprocessed data
        use_heavy_aug: Whether to use heavily augmented data
    """
    print("\n" + "="*80)
    print("Loading T5 data for SCRATCH training...")
    print(f"Schema context: {use_schema}")
    print(f"Preprocessed data: {use_preprocessed}")
    print(f"Heavy augmentation: {use_heavy_aug}")
    print("="*80)
    
    train_loader = get_dataloader_scratch(batch_size, "train", use_schema, use_preprocessed, use_heavy_aug)
    dev_loader = get_dataloader_scratch(test_batch_size, "dev", use_schema, use_preprocessed, False)  # No aug for dev
    test_loader = get_dataloader_scratch(test_batch_size, "test", use_schema, use_preprocessed, False)  # No aug for test
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("="*80 + "\n")
    
    return train_loader, dev_loader, test_loader