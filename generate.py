import argparse

import torch

from models.model import BigramLanguageModel
from utils.config import load_config
from utils.data_utils import create_vocab, decode
from utils.evaluation_utils import generate_text

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

# Initialize model
model = BigramLanguageModel(vocab_size, n_embd, n_head,
                            n_layer, block_size, dropout).to(device)

# Load the best model checkpoint
checkpoint_path = 'checkpoints/best_model.pth'  # TODO: feat: specify model from cli
model.load_state_dict(torch.load(checkpoint_path, map_location=device))  # map_location=torch.device('cpu')

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate text with different temperatures.')
parser.add_argument('--temperature', nargs='+', type=float, default=[1.0],
                    help='List of temperatures to use for text generation.')
args = parser.parse_args()

# Generate text for each temperature
context = torch.zeros((1, 1), dtype=torch.long, device=device)
for temp in args.temperature:
    generated_text = generate_text(model, context, max_new_tokens=1000, temperature=temp,
                                   block_size=block_size, device=device)
    print("\n#######################################")
    print(f'Generated text with temperature of: {temp}')
    print("#######################################")
    print(decode(generated_text[0].tolist(), itos))
