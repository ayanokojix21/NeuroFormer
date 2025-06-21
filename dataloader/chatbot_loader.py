
# Importing Libraries
import torch
from torch.utils.data import Dataset, DataLoader

# Defining Dataset Class
class ChatDataset(Dataset):
    
    def __init__(self, path, block_size):
        self.data = torch.load(path)
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        input_ids = self.data[idx: idx + self.block_size]
        labels = self.data[idx + 1: idx + self.block_size + 1]
        return {
            'input_ids' : input_ids,
            'labels' : labels
        }
        
# Defining DataLoader Function to load dataset
def get_chat_dataloader(path, batch_size=4, block_size=256, shuffle=True):
    dataset = ChatDataset(path, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)