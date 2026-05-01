import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("Loading original clean.csv data...")
df_clean = pd.read_csv('clean.csv')
df_clean = df_clean.dropna(subset=['text', 'label'])
df_clean['text'] = df_clean['text'].astype(str)

print("Loading additional sample_data datasets...")
df_cnn = pd.read_excel('sample_data/cleaned/dataset_cnn_10k_cleaned.xlsx')
df_kompas = pd.read_excel('sample_data/cleaned/dataset_kompas_4k_cleaned.xlsx')
df_tempo = pd.read_excel('sample_data/cleaned/dataset_tempo_6k_cleaned.xlsx')
df_hoax = pd.read_excel('sample_data/cleaned/dataset_turnbackhoax_10_cleaned.xlsx')

df_cnn_data = pd.DataFrame({'text': df_cnn['text_new'].astype(str), 'label': df_cnn['hoax']})
df_kompas_data = pd.DataFrame({'text': df_kompas['text_new'].astype(str), 'label': df_kompas['hoax']})
df_tempo_data = pd.DataFrame({'text': df_tempo['text_new'].astype(str), 'label': df_tempo['hoax']})
df_hoax_data = pd.DataFrame({'text': df_hoax['Title'].astype(str) + " " + df_hoax['FullText'].astype(str), 'label': df_hoax['hoax']})

df = pd.concat([df_clean, df_cnn_data, df_kompas_data, df_tempo_data, df_hoax_data], ignore_index=True)
df = df.dropna(subset=['text', 'label'])

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

X = df['text'].values
y = df['label'].values

print("Tokenizing data...")
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 300

# Build vocabulary
all_words = ' '.join(X).split()
word_counts = Counter(all_words)
common_words = word_counts.most_common(MAX_VOCAB_SIZE - 2)
word2idx = {word: idx + 2 for idx, (word, _) in enumerate(common_words)}
word2idx['<PAD>'] = 0
word2idx['<UNK>'] = 1

def text_to_sequence(text, word2idx, max_len):
    words = text.split()
    seq = [word2idx.get(w, 1) for w in words]
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

X_seq = np.array([text_to_sequence(t, word2idx, MAX_SEQUENCE_LENGTH) for t in X])

X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define LSTM Model
class HoaxLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, drop_prob=0.3):
        super(HoaxLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Take the output from the last time step
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = HoaxLSTM(
    vocab_size=MAX_VOCAB_SIZE, 
    embed_dim=100, 
    hidden_dim=64, 
    output_dim=1, 
    n_layers=2
).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training model...")
EPOCHS = 3
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            val_loss += criterion(output, labels).item()
            
            predicted = (output > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
    val_loss /= len(test_loader)
    val_acc = correct / total
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'hoax_lstm_model.pth')

print("Saving tokenizer...")
with open('word2idx.pickle', 'wb') as handle:
    pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Training complete! Model saved as 'hoax_lstm_model.pth' and tokenizer as 'word2idx.pickle'")
