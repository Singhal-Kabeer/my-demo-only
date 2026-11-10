import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CJPEDataset(Dataset):
    def __init__(self, data_list, max_len=1024):
        self.input_ids = []
        self.labels = []        
        for doc in data_list:
            encoded_doc = doc['tokens']
            truncated_doc = encoded_doc[-max_len:]
            self.input_ids.append(truncated_doc)
            self.labels.append(doc['label'])
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return inputs_padded, labels