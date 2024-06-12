import torch

from models.model import BigramLanguageModel
from utils.config import load_config
from utils.data_utils import create_vocab, load_data
from utils.training_utils import estimate_loss

# Reproducibility
torch.manual_seed(1337)

# Load configuration
config = load_config('configs/version_2.yaml')  # TODO: feat: specify configs from cli

# Hyperparameters and device setup
batch_size = config['batch_size']
block_size = config['block_size']
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


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# Initialize model
model = BigramLanguageModel(vocab_size, config['n_embd'], config['n_head'],
                            config['n_layer'], block_size, config['dropout']).to(device)

# Load the best model checkpoint
checkpoint_path = 'checkpoints/version_2.pth'  # TODO: feat: specify model from cli
model.load_state_dict(torch.load(checkpoint_path))

# Evaluate the model
losses = estimate_loss(model, eval_iters, get_batch)
print(f"Validation loss: {losses['val']:.4f}")
