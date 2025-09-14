import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from sklearn.model_selection import KFold

class MoltenpoolDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 transform=None, 
                 mode='train', 
                 k_fold=5, 
                 current_fold=0, 
                 seed=42, 
                 test_ratio=0.2):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode.lower()
        self.k_fold = k_fold
        self.current_fold = current_fold
        self.seed = seed
        self.test_ratio = test_ratio
        self.loaded_num = 4  # Sequence length
        self.box = (195, 50, 465, 320)  # Crop region
        
        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Prepare sequences
        self.sequences = self._prepare_sequences()
        
        # Split data
        self.data = self._split_data()

    def _prepare_sequences(self):
        exp_data = {}
        
        for img_name in os.listdir(self.root_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                parts = img_name.split('_')
                if len(parts) < 4:
                    continue
                    
                try:
                    exp_id, frame_id, power, label = parts[0], parts[1], parts[2], parts[3]
                    label = label.split('.')[0]
                    
                    img_path = os.path.join(self.root_dir, img_name)
                    
                    if exp_id not in exp_data:
                        exp_data[exp_id] = []
                        
                    exp_data[exp_id].append({
                        'frame_id': frame_id,
                        'img_path': img_path,
                        'power': power,
                        'label': label,
                    })
                except (IndexError, ValueError):
                    continue
        
        # Create n-frame sequences
        n = 15
        sequences = []
        for exp_id, frames in exp_data.items():
            frames.sort(key=lambda x: int(x['frame_id']))
            
            for i in range(0, len(frames), n):
                sequence = frames[i:i+n]
                sequences.append({
                    'exp_id': exp_id,
                    'frames': sequence,
                    'sequence_idx': i // n
                })
        
        return sequences

    def _split_data(self):
        # Shuffle sequences
        random.shuffle(self.sequences)
        
        # Split test set
        test_size = int(len(self.sequences) * self.test_ratio)
        test_sequences = self.sequences[:test_size]
        remaining_sequences = self.sequences[test_size:]
        
        # Split remaining for K-fold
        if self.k_fold > 1 and self.mode != 'test':
            kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=self.seed)
            folds = list(kf.split(remaining_sequences))
            train_seq_idx, val_seq_idx = folds[self.current_fold]
            train_sequences = [remaining_sequences[i] for i in train_seq_idx]
            val_sequences = [remaining_sequences[i] for i in val_seq_idx]
        else:
            val_size = int(len(remaining_sequences) * 0.2)
            val_sequences = remaining_sequences[:val_size]
            train_sequences = remaining_sequences[val_size:]
        
        # Select sequences based on mode
        if self.mode == 'train':
            selected_sequences = train_sequences
        elif self.mode == 'val':
            selected_sequences = val_sequences
        elif self.mode == 'test':
            selected_sequences = test_sequences
        else:
            raise ValueError(f"Mode '{self.mode}' not supported. Use 'train', 'val', or 'test'.")
        
        # Create sliding windows
        windowed_samples = []
        for seq in selected_sequences:
            frames = seq['frames']
            
            # Create windows within same sequence
            if len(frames) >= self.loaded_num:
                for i in range(len(frames) - self.loaded_num + 1):
                    window_frames = frames[i:i + self.loaded_num]
                    
                    try:
                        window_label = float(window_frames[-1]['label'])
                    except ValueError:
                        continue
                    
                    windowed_samples.append({
                        'exp_id': seq['exp_id'],
                        'sequence_idx': seq['sequence_idx'],
                        'frames': window_frames,
                        'label': window_label,
                        'window_start_idx': i
                    })
        
        return windowed_samples

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        info = self.data[idx]
        frame_items = info['frames']
        label = info['label']
        
        try:
            power = float(frame_items[-1]['power'])
        except (ValueError, TypeError):
            power = 1.0
        
        image_set = []
        for item in frame_items:
            image_path = item['img_path']
            
            try:
                with Image.open(image_path) as image:
                    image = image.convert("L").crop(self.box)
                    if self.transform:
                        image = self.transform(image)
                    image_set.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        
        image_seq = torch.stack(image_set)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        power_tensor = torch.tensor(power, dtype=torch.float32).unsqueeze(0)
        
        return image_seq, label_tensor, power_tensor