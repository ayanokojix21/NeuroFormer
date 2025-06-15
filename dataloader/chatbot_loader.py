# Importing Libraries
import torch
from torch.utils.data import Dataset, DataLoader

# Defining Dataset Class
class ChatDataset(Dataset):
    
    def __init__(self, path):
        self.data = torch.load(path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids' : self.data[idx]['input_ids'],
            'attention_mask' : self.data[idx]['attention_mask']
        }
        
# Defining DataLoader Function to load dataset
def get_chat_dataloader(path, batch_size=4, shuffle=True, num_workers=2, pin_memory=True):
    dataset = ChatDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)