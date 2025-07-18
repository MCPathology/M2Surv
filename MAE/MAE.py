import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

def complete_masking(batch, p, n_tokens):

    padding_token = 1
    cls_token = 3

    indices = batch['tokenized_gene']

    indices = torch.where(indices == 0, torch.tensor(padding_token), indices) # 0 is originally the padding token, we change it to 1
    batch['tokenized_gene'] = indices

    mask = 1 - torch.bernoulli(torch.ones_like(indices), p) # mask indices with probability p
    spatial_mask = 1 - torch.bernoulli(torch.ones_like(indices), 1)

    masked_indices = indices * mask # masked_indices 
    masked_indices = torch.where(indices != padding_token, masked_indices, indices) # we just mask non-padding indices
    mask = torch.where(indices == padding_token, torch.tensor(padding_token), mask) # in the model we evaluate the loss of mask position 0
    spatial_mask = torch.where(indices == padding_token, torch.tensor(padding_token), spatial_mask) # in the model we evaluate the loss of mask position 0
    # so we make the mask of all PAD tokens to be 1 so that it's not taken into account in the loss computation

    # Notice for the following 2 lines that masked_indices has already not a single padding token masked
    masked_indices = torch.where(indices != cls_token, masked_indices, indices) # same with CLS, no CLS token can be masked
    mask = torch.where(indices == cls_token, torch.tensor(padding_token), mask) # we change the mask so that it doesn't mask any CLS token
    spatial_mask = torch.where(indices == cls_token, torch.tensor(padding_token), spatial_mask) # we change the mask so that it doesn't mask any CLS token

    # 80% of masked indices are masked
    # 10% of masked indices are a random token
    # 10% of masked indices are the real token

    random_tokens = torch.randint(10, n_tokens, size=masked_indices.shape, device=masked_indices.device)
    random_tokens = random_tokens * torch.bernoulli(torch.ones_like(random_tokens)*0.1).type(torch.int64) 

    masked_indices = torch.where(masked_indices == 0, random_tokens, masked_indices) # put random tokens just in the previously masked tokens

    same_tokens = indices.clone()
    same_tokens = same_tokens * torch.bernoulli(torch.ones_like(same_tokens) * 0.1).type(torch.int64)

    masked_indices = torch.where(masked_indices == 0, same_tokens, masked_indices) # put same tokens just in the previously masked tokens

    batch['masked_indices'] = masked_indices
    batch['mask'] = mask
    batch['spatial_mask'] = spatial_mask
    attention_mask = (masked_indices == padding_token)
    batch['attention_mask'] = attention_mask.type(torch.bool)

    return batch

class GDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'tokenized_gene': self.data[idx]}

    def load_gene_data_from_csv(file_path, gene_column):
        df = pd.read_csv(file_path)
    
        gene_data = df[gene_column].tolist()
    
        tokenized_gene_data = [list(map(int, gene.split(','))) for gene in gene_data]
    
        return tokenized_gene_data

class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=10):
        model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
        
            masked_batch = complete_masking(batch, p=0.15, n_tokens=100)
            inputs = masked_batch['masked_indices'].float()  

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

dataset = ExampleDataset(data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

input_dim = 10 
hidden_dim = 256 
model = MaskedAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_autoencoder(model, dataloader, criterion, optimizer)