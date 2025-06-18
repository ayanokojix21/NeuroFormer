# Importing Libraries
import torch
from torch.utils.data import Dataset, DataLoader

# Defining Dataset Class
class ChatDataset(Dataset):
    
    def __init__(self, path):
        self.data = torch.load(path)
        
    def __len__(self):
        return len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids' : self.data['input_ids'][idx],
            'attention_mask' : self.data['attention_mask'][idx]
        }
        
# Defining DataLoader Function to load dataset
def get_chat_dataloader(path, batch_size=4, shuffle=True):
    dataset = ChatDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)