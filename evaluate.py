import torch

from models.model import BigramLanguageModel
from utils.config import load_config
from utils.data_utils import create_vocab, get_batch, load_data
from utils.training_utils import estimate_loss

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


# Initialize model
model = BigramLanguageModel(vocab_size, n_embd, n_head,
                            n_layer, block_size, dropout).to(device)

# Load the best model checkpoint
checkpoint_path = 'checkpoints/best_model.pth'  # TODO: feat: specify model from cli
model.load_state_dict(torch.load(checkpoint_path, map_location=device))  # map_location=torch.device('cpu')

# Evaluate the model
losses = estimate_loss(model, eval_iters, get_batch_split)
print(f"Validation loss: {losses['val']:.4f}")
