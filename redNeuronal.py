import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MiModelo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(MiModelo, self).__init__()
        # Capas del modelo
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, texto):
        #Propagacion de datos a travez de capas de modelo
        embedded = self.embedding(texto)
        transformer_out = self.transformer(embedded)
        pooled = transformer_out.mean(dim=0)
        output = self.fc(pooled)
        return output

class MiDataset(Dataset):
    def __init__(self, text):
        self.text = text.split()
        self.vocab = set(self.text)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        self.data = [self.word_to_idx[word] for word in self.text]
        
    def __len__(self):
        return len(self.data) - sequence_length

    def __getitem__(self, idx):
        inputs = torch.LongTensor(self.data[idx:idx+sequence_length])
        targets = torch.LongTensor([self.data[idx+sequence_length]])
        return inputs, targets

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = torch.stack(targets, dim=1).squeeze()
    return inputs, targets

# Hiperparametros del modelo
vocab_size = 10000
embedding_dim = 128
output_dim = 256
lr = 0.001
num_epochs = 10
batch_size = 32
sequence_length = 10

# Datos de entrenamiento
train_text = ... # Aquí coloca tus datos de entrenamiento
train_dataset = MiDataset(train_text)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Datos de prueba
test_text = ... # Aquí coloca tus datos de prueba
test_dataset = MiDataset(test_text)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Entrenamiento modelo
modelo = MiModelo(vocab_size, embedding_dim, output_dim)
optimizer = optim.Adam(modelo.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    epoch_loss = 0
    modelo.train()
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = modelo(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print('Epoch {}/{} - Training Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss / len(train_dataloader)))

    # Evaluación del modelo en el conjunto de datos de prueba
    modelo.eval()
    correct =
