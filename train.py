import torch
import torch.optim as optim

from models.model import BigramLanguageModel
from utils.config import load_config
from utils.data_utils import create_vocab, get_batch, load_data
from utils.training_utils import estimate_loss, train

# Reproducibility
torch.manual_seed(1337)

# Load configuration
config = load_config('configs/version_2.yaml')  # TODO: feat: specify configs from cli

# Hyperparameters and device setup
batch_size = config['batch_size']
block_size = config['block_size']
max_iters = config['max_iters']
eval_interval = config['eval_interval']
learning_rate = config['learning_rate']
n_embd = config['n_embd']
n_head = config['n_head']
n_layer = config['n_layer']
dropout = config['dropout']
eval_iters = config['eval_iters']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load and preprocess data
with open('./data/tiny_shakespeare/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create vocabulary
vocab_size, stoi, itos = create_vocab(text)

# Load data
data = load_data('./data/tiny_shakespeare/input.txt', stoi)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch_split(split):
    data = train_data if split == 'train' else val_data
    return get_batch(data, batch_size, block_size, device)


# Initialize model and optimizer
model = BigramLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training
print('##### Training started #####')
train_losses = []
best_validation_loss = float('inf')
model, train_losses = train(model, optimizer, get_batch_split, estimate_loss, max_iters,
                            eval_interval, best_validation_loss, train_losses, eval_iters)

# Save the model
torch.save(model.state_dict(), 'checkpoints/best_model.pth')
print('Model saved to checkpoints/best_model.pth')
print('##### Training complete #####')
