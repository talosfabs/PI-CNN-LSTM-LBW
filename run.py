import argparse
import logging
import torch 
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from model import CnnLstmPinnModel 
from utils import set_seed, get_optimizer, get_scheduler
from trainer import Trainer
from dataset.moltenpool_dataset import MoltenpoolDataset

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    train_augs = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1719], std=[0.0315])
    ])

    n_splits = args.k_fold
    all_fold_test_metrics = []

    for fold in range(n_splits):
        output_path = f'./weights/fold_{fold+1}'
        os.makedirs(output_path, exist_ok=True)

        train_dataset = MoltenpoolDataset(
            root_dir=args.data_path,
            transform=train_augs,
            mode='train',
            k_fold=n_splits,
            current_fold=fold,
            seed=args.seed
        )
            
        val_dataset = MoltenpoolDataset(
            root_dir=args.data_path,
            transform=train_augs,
            mode='val',
            k_fold=n_splits,
            current_fold=fold,
            seed=args.seed
        )
    
        test_dataset = MoltenpoolDataset(
            root_dir=args.data_path,
            transform=train_augs,
            mode='test',
            k_fold=n_splits,
            current_fold=fold,
            seed=args.seed
        )
    
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
        logging.info(f'fold {fold+1}\ttrain set:{len(train_dataset)}\tvalidation set:{len(val_dataset)}\ttest set:{len(test_dataset)}')
    
        model = CnnLstmPinnModel()
        trainer = Trainer(args.epochs, model, device, train_augs)
        optimizer = get_optimizer(trainer.model, args.optim_type, args.lr)
        scheduler = get_scheduler(optimizer, args.use_scheduler, args.epochs, args.lr, train_loader)

        best_val_mae = float('inf')
        best_epoch = -1
        best_test_metrics = None

        for epoch in range(args.epochs):
            train_loss, train_metrics = trainer.train_epoch(train_loader, optimizer, scheduler, epoch, args.epochs)
            val_loss, val_metrics = trainer.evaluate_epoch(val_loader, task_name="valid")  
            
            logging.info(f"Epoch {epoch+1}/{args.epochs}:")
            logging.info(f"  Train - Loss: {train_loss:.4f}, MAE: {train_metrics.get('MAE', 0):.4f}, MSE: {train_metrics.get('MSE', 0):.4f}, RMSE: {train_metrics.get('RMSE', 0):.4f}")
            logging.info(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_metrics.get('MAE', 0):.4f}, MSE: {val_metrics.get('MSE', 0):.4f}, RMSE: {val_metrics.get('RMSE', 0):.4f}")
            
            current_mae = val_metrics.get('MAE', val_loss)
            
            if current_mae < best_val_mae:
                best_val_mae = current_mae
                best_epoch = epoch
                trainer.save_checkpoint(output_path, fold)

        if trainer.get_model_path() and os.path.exists(trainer.get_model_path()):
            try:
                trainer.load_model()
                test_loss, test_metrics = trainer.evaluate_epoch(test_loader, task_name="test")  
                logging.info(f"  Test  - Loss: {test_loss:.4f}, MAE: {test_metrics.get('MAE', 0):.4f}, MSE: {test_metrics.get('MSE', 0):.4f}, RMSE: {test_metrics.get('RMSE', 0):.4f}")
                
                best_test_metrics = test_metrics
            except Exception as e:
                logging.error(f"Error loading or evaluating model: {str(e)}")

        if best_test_metrics:
            all_fold_test_metrics.append(best_test_metrics)
            
        logging.info(f"Fold {fold} completed. Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch+1}")
        
    if all_fold_test_metrics:
        avg_metrics = {}
        std_metrics = {}
        
        metric_keys = set()
        for metrics in all_fold_test_metrics:
            metric_keys.update(metrics.keys())
        
        for key in metric_keys:
            values = [metrics.get(key, 0) for metrics in all_fold_test_metrics]
            avg_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
        
        logging.info("All folds completed. Test metrics summary:")
        for key in metric_keys:
            logging.info(f"  {key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")

def get_args():
    parser = argparse.ArgumentParser(description='Train the PiCnnLstm')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3, help='Learning rate', dest='lr')
    parser.add_argument('--k_fold', '-kf', metavar='KF', type=int, default=5, help='Cross validation folders')
    parser.add_argument('--data-path', '-dp', dest='data_path', type=str, default='./data', help='Data path')
    parser.add_argument('--test-ratio', '-tr', dest='test_ratio', type=float, default=0.2, help='Test data ratio')
    parser.add_argument('--optimizer', '-op', dest='optim_type', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--use-scheduler', '-lr_sch', dest='use_scheduler', type=bool, default=True, help='Use learning rate scheduler')
    parser.add_argument('--random-seed', '-seed', dest='seed', type=int, default=42, help='Random seed')

    return parser.parse_args()

def main():
    args = get_args()
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f'Configuration: {vars(args)}')
    
    try:
        train(args)
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main()