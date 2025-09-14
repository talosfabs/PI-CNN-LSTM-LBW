import torch
import torch.optim as optim
from typing import Optional, Any
from pathlib import Path

def save_checkpoint(filepath: str, model: torch.nn.Module) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict()
    }, filepath)

def get_optimizer(model: torch.nn.Module, optim_type: str,lr: float, weight_decay=1e-4) -> torch.optim.Optimizer:
    available_optimizers = [name for name in dir(optim) 
                            if not name.startswith('_') and 
                            hasattr(getattr(optim, name), '__call__')]

    optimizer_class = getattr(optim, optim_type)

    return optimizer_class(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )

def get_scheduler(optimizer: torch.optim.Optimizer,
                 use_scheduler: bool,
                 epochs: int,
                 lr: float,
                 dataloader: Optional[torch.utils.data.DataLoader] = None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if use_scheduler and dataloader is not None:
        total_steps = len(dataloader) * epochs
        print(f"Initializing OneCycleLR scheduler with total_steps: {total_steps}")
        
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps
        )
    
    return None