import torch

# Load the tokenized data
data = torch.load("data/tokenized/Translation Data/train.pt")

# Check the shape
print("Shape of input_ids:", data['input_ids'].shape)

print("Max input_ids:", data["input_ids"].max().item())
print("Max labels:", data["labels"].max().item())
print("Any input_ids >= vocab_size?", (data["input_ids"] >= 16384).any().item())
print("Any labels >= vocab_size?", (data["labels"] >= 16384).any().item())
print("Any input_ids < 0?", (data["input_ids"] < 0).any().item())
print("Any labels < -1?", (data["labels"] < -1).any().item())  # valid label is -100 for ignored loss
