"""
A module containing NLP models and utilities for text processing and classification.

This module provides:
- Text data processing and dataset handling
- Model architectures (Transformer, MLP)
- Dataset classes for text data
- Model building utilities
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import OrderedDict

# =============================================================================
# Model Architectures
# =============================================================================

class TransformerLayer(nn.Module):
    """A single transformer layer with multi-head attention and feed-forward network.
    
    Args:
        hidden_size (int): Size of hidden states
        num_attention_heads (int): Number of attention heads
        intermediate_size (int): Size of intermediate layer in feed-forward network
    """
    def __init__(self, hidden_size=256, num_attention_heads=4, intermediate_size=1024):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Forward pass through the transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class EncoderTransformer(nn.Module):
    """A transformer-based model for text classification using encoder-only architecture.
    
    Args:
        vocab_size (int): Size of vocabulary
        num_classes (int): Number of output classes
        num_layers (int): Number of transformer layers
        hidden_size (int): Size of hidden states
    """
    def __init__(self, vocab_size, num_classes, num_layers=4, hidden_size=256):
        super().__init__()
        self.vocab_size = vocab_size  # Store vocab size for pruning
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Forward pass through the encoder transformer model.
        
        Args:
            x (torch.Tensor): Input tensor of token IDs
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = self.embedding(x).permute(1, 0, 2)
        x = self.dropout(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.mean(dim=0)
        return self.classifier(x)


class MLP(nn.Module):
    """A flexible multi-layer perceptron for text classification, with named layers.
    
    Args:
        input_dim (int): Size of the vocabulary (for the embedding).
        hidden_dims (list of int): List of hidden layer sizes.
        num_classes (int): Number of output classes.
        embedding_dim (int, optional): Dimension of word embeddings (default: 128).
    """
    def __init__(self, input_dim, hidden_dims, num_classes, embedding_dim=128):
        super().__init__()
        self.vocab_size = input_dim  # Store vocab size for pruning
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.n_hidden = len(hidden_dims)
        self.activation = nn.ReLU()
        
        layers = OrderedDict()
        prev_dim = embedding_dim
        
        # build hidden layers with explicit names
        for i, hidden_dim in enumerate(hidden_dims, start=1):
            setattr(self, f'fc{i}', nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # final output layer
        setattr(self, f'fc{self.n_hidden + 1}', nn.Linear(prev_dim, num_classes))
        
        self.layers = nn.Sequential(layers)
        
    def forward(self, x):
        # x: LongTensor of shape (batch, seq_len)
        x = self.embedding(x)           # -> (batch, seq_len, embedding_dim)
        x = x.mean(dim=1)  
        for i in range(1, self.n_hidden + 1):
            x = getattr(self, f'fc{i}')(x)
            x = self.activation(x)
        x = getattr(self, f'fc{self.n_hidden + 1}')(x)# mean over seq_len -> (batch, embedding_dim)
        return x

class EncoderDecoderTransformer(nn.Module):
    """
    A transformer-based model for text classification using encoder-decoder architecture.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of word embeddings
        num_classes (int): Number of output classes
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int, 
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super(EncoderDecoderTransformer, self).__init__()
        self.vocab_size = vocab_size  # Store vocab size for pruning
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.embedding(x)
        x = self.transformer(x, x)  # Using same tensor for source and target
        x = torch.mean(x, dim=1)    # Global average pooling
        x = self.fc(x)
        return x

class BagOfWordsMLP(nn.Module):
    """A multi-layer perceptron for text classification using bag-of-words representation.
    
    Args:
        vocab_size (int): Size of the vocabulary (input dimension)
        hidden_dims (list of int): List of hidden layer sizes
        num_classes (int): Number of output classes
        dropout (float, optional): Dropout rate (default: 0.1)
    """
    def __init__(self, vocab_size, hidden_dims, num_classes, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size  # Store vocab size for pruning
        self.n_hidden = len(hidden_dims)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Input layer (vocab_size -> first hidden dim)
        self.fc1 = nn.Linear(vocab_size, hidden_dims[0])
        
        # Hidden layers
        for i in range(1, self.n_hidden):
            setattr(self, f'fc{i+1}', nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dims[-1], num_classes)
        
    def forward(self, x):
        # x: LongTensor of shape (batch, seq_len)
        batch_size = x.size(0)
        bow = torch.zeros(batch_size, self.fc1.in_features, device=x.device)
        
        # Convert each sequence to bag of words using bincount
        for i in range(batch_size):
            # Count occurrences of each token in the sequence
            counts = torch.bincount(x[i], minlength=self.fc1.in_features)
            bow[i] = counts.float()
        
        # Forward pass through layers
        x = self.fc1(bow)
        x = self.activation(x)
        x = self.dropout(x)
        
        for i in range(1, self.n_hidden):
            x = getattr(self, f'fc{i+1}')(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.fc_out(x)
        return x

# =============================================================================
# Model Building Functions
# =============================================================================

def build_encoder_decoder_transformer(config: Dict[str, Any]) -> EncoderDecoderTransformer:
    """
    Build an encoder-decoder transformer model based on the provided configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - vocab_size (int): Size of the vocabulary
            - embedding_dim (int): Dimension of word embeddings
            - num_classes (int): Number of output classes
            - num_layers (int, optional): Number of transformer layers (default: 2)
            - num_heads (int, optional): Number of attention heads (default: 4)
            - dropout (float, optional): Dropout rate (default: 0.1)
            
    Returns:
        EncoderDecoderTransformer: Initialized transformer model
    """
    required_params = ['vocab_size', 'embedding_dim', 'num_classes']
    for param in required_params:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
    
    return EncoderDecoderTransformer(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        num_classes=config['num_classes'],
        num_layers=config.get('num_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1)
    )

def build_encoder_transformer(config: Dict[str, Any]) -> EncoderTransformer:
    """
    Build an encoder-only transformer model based on the provided configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - vocab_size (int): Size of vocabulary
            - num_classes (int): Number of output classes
            - hidden_size (int): Size of hidden states
            - num_layers (int): Number of transformer layers
            
    Returns:
        EncoderTransformer: Initialized transformer model
    """
    required_params = ['vocab_size', 'num_classes', 'hidden_size', 'num_layers']
    for param in required_params:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
    
    return EncoderTransformer(
        vocab_size=config['vocab_size'],
        num_classes=config['num_classes'],
        num_layers=config['num_layers'],
        hidden_size=config['hidden_size']
    )

def build_mlp_model_with_embeddings(config: Dict[str, Any]) -> MLP:
    """
    Build an MLP model based on the provided configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - input_dim (int): Input dimension (vocabulary size)
            - num_classes (int): Number of output classes
            - hidden_dims (list): List of hidden layer dimensions
            - embedding_dim (int, optional): Dimension of word embeddings (default: 128)
            
    Returns:
        MLP: Initialized MLP model
    """
    required_params = ['input_dim', 'num_classes', 'hidden_dims']
    for param in required_params:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
    
    return MLP(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        num_classes=config['num_classes'],
        embedding_dim=config.get('embedding_dim', 128)
    )

def build_pretrained_transformer(model_name: str, num_classes: int):
    """
    Load a pretrained HuggingFace transformer model and tokenizer for sequence classification.
    Args:
        model_name (str): Name or path of the pretrained model (e.g., 'bert-base-uncased')
        num_classes (int): Number of output classes
    Returns:
        model: HuggingFace transformer model
        tokenizer: Corresponding tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    return model, tokenizer

def build_bow_mlp_model(config: Dict[str, Any]) -> BagOfWordsMLP:
    """Build a bag-of-words MLP model based on the provided configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - vocab_size (int): Size of the vocabulary
            - hidden_dims (list of int): List of hidden layer sizes
            - num_classes (int): Number of output classes
            - dropout (float, optional): Dropout rate (default: 0.1)
            
    Returns:
        BagOfWordsMLP: Initialized bag-of-words MLP model
    """
    required_params = ['vocab_size', 'hidden_dims', 'num_classes']
    for param in required_params:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
    
    return BagOfWordsMLP(
        vocab_size=config['vocab_size'],
        hidden_dims=config['hidden_dims'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.1)
    )

# Model configurations
ENCODER_DECODER_CONFIGS = {
    'small': {
        'embedding_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1
    },
    'medium': {
        'embedding_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1
    },
    'large': {
        'embedding_dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1
    },    
    'Ksenia': {
        'embedding_dim': 128,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0
    }
}

ENCODER_CONFIGS = {
    'small': {
        'hidden_size': 128,
        'num_layers': 2
    },
    'medium': {
        'hidden_size': 256,
        'num_layers': 4
    },
    'large': {
        'hidden_size': 512,
        'num_layers': 6
    }
}

MLP_CONFIGS = {
    'small': {
        'hidden_dims': [256, 128],
        'embedding_dim': 128
    },
    'medium': {
        'hidden_dims': [512, 256, 128],
        'embedding_dim': 256
    },
    'large': {
        'hidden_dims': [1024, 512, 256, 128],
        'embedding_dim': 512
    },
    'Ksenia': {
        'hidden_dims': [512, 256],
        'embedding_dim': 128,
    }
}

BOW_MLP_CONFIGS = {
    'small': {
        'hidden_dims': [256, 128],
        'dropout': 0.1
    },
    'medium': {
        'hidden_dims': [512, 256, 128],
        'dropout': 0.1
    },
    'large': {
        'hidden_dims': [1024, 512, 256, 128],
        'dropout': 0.1
    },
    'Ksenia': {
        'hidden_dims': [128,512, 256],
        'dropout': 0
    }
}

def build_nlp_model(vocab_size, num_classes, model_type='encoder_decoder', size='medium', pretrained_model_name=None):
    """Build a model of specified type and size, with optional support for pretrained transformers.
    Args:
        vocab_size (int): Size of vocabulary (ignored for pretrained transformers)
        num_classes (int): Number of output classes
        model_type (str): Type of model ('encoder_decoder', 'encoder', 'mlp', 'bow_mlp', or 'pretrained_transformer')
        size (str): Model size ('small', 'medium', or 'large')
        pretrained_model_name (str, optional): Name or path of pretrained transformer (if using)
    Returns:
        nn.Module or (nn.Module, tokenizer): Configured model (and tokenizer if pretrained)
    """
    if model_type == 'pretrained_transformer':
        if pretrained_model_name is None:
            raise ValueError('pretrained_model_name must be specified for pretrained_transformer')
        model, tokenizer = build_pretrained_transformer(pretrained_model_name, num_classes)
        return model, tokenizer
    if model_type == 'encoder_decoder':
        config = ENCODER_DECODER_CONFIGS[size].copy()
        config.update({
            'vocab_size': vocab_size,
            'num_classes': num_classes
        })
        return build_encoder_decoder_transformer(config)
    elif model_type == 'encoder':
        config = ENCODER_CONFIGS[size].copy()
        config.update({
            'vocab_size': vocab_size,
            'num_classes': num_classes
        })
        return build_encoder_transformer(config)
    elif model_type == 'mlp_with_embeddings':
        config = MLP_CONFIGS[size].copy()
        config.update({
            'input_dim': vocab_size,
            'num_classes': num_classes
        })
        return build_mlp_model_with_embeddings(config)
    elif model_type == 'bow_mlp':
        config = BOW_MLP_CONFIGS[size].copy()
        config.update({
            'vocab_size': vocab_size,
            'num_classes': num_classes
        })
        return build_bow_mlp_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'encoder_decoder', 'encoder', 'mlp', 'bow_mlp', or 'pretrained_transformer'") 