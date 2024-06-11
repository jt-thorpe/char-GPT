# Character-Level GPT Model

## Project Description

This project is a character-level GPT (Generative Pre-trained Transformer) model, inspired by Andrej Karpathy's "Neural Nets: Zero to Hero" series. It was developed as a learning project to understand transformer models and to experiment with generating text at the character level.

The model is trained on a dataset of Shakespeare's works and is capable of generating new text in a similar style. It uses a simple, modular design to make the architecture easy to understand and extend.

## Features

- **Character-Level Language Modeling**: The model predicts the next character in a sequence, allowing it to generate text that looks a little like Shakespeare.
- **Transformer Architecture**: Implements the transformer architecture with multi-head self-attention and feed-forward networks.
- **Configurable Hyperparameters**: Easily adjustable hyperparameters via a YAML configuration file.
- **Training, Evaluation, and Text Generation Scripts**: Separate scripts for training the model, evaluating its performance, and generating new text.
- **Customisability**: You can of course edit whatever you wish and make this your own!

## Installation

I assume some knowledge of how to set-up a venv and install dependencies etc, but generally it goes like:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/char-gpt.git
    cd char-gpt
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv_name
    source venv_name/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the model, run the training script:
```bash
python train.py
```

### Evaluating the Model

To evaluate the model on the validation set, run the evaluation script:

```bash
python evaluate.py
```

### Generating Text

To generate text using the trained model, run the generate script:

```bash
python generate.py
```

By default, a temperature of 1.0 is used when generating text. You can specify as many temperatures as you like such as:

```bash
python generate.py --temperatures 0.5 0.7 1.25
```

### Configuring the Model

The model's hyperparameters and other settings can be configured via the config.yaml file located in the configs directory. Here is an example of the configuration file:

```YAML
batch_size: 64
block_size: 256
max_iters: 10000
eval_interval: 250
learning_rate: 0.0003
n_embd: 384
n_head: 6
n_layer: 6
dropout: 0.2
eval_iters: 20
```

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Inspired by Andrej Karpathy's "Neural Networks: Zero to Hero" series
- Based on the transformer architecture described in the paper "Attention Is All You Need" by Vaswani et al.