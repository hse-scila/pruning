"""
A module for experiment setup, training, and dataset handling.

This module provides:
- Dataset loading and preprocessing
- Model training and evaluation
- Experiment setup and execution utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torchvision
import torchvision.transforms as transforms
from nlp_models import build_nlp_model
from IPython import display
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_cifar_data(dataset_name: str = 'cifar10', data_dir: str = './data', 
                   batch_size: int = 32, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 or CIFAR-100 dataset with appropriate transforms.
    
    Args:
        dataset_name (str): Name of the dataset ('cifar10' or 'cifar100')
        data_dir (str): Directory to store the dataset
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load dataset
    if dataset_name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test)
    elif dataset_name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create dataloaders
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_imdb_data(data_dir):
    """Load IMDB dataset from directory structure.
    
    Args:
        data_dir (str): Directory containing 'neg' and 'pos' subdirectories
        
    Returns:
        pd.DataFrame: DataFrame with 'text' and 'label' columns
    """
    texts = []
    labels = []

    for label, folder in enumerate(['neg', 'pos']):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                texts.append(text)
                labels.append(label)
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df

def load_reviews_data(csv_path):
    """Load reviews dataset from CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        text_col (str): Name of text column
        label_col (str): Name of label column
        
    Returns:
        tuple: (train_df, test_df) - Preprocessed pandas DataFrames with 'text' and 'label' columns
    """
    # Load data
    df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip', low_memory=False)
    tt = ['Score', 'Text']
    df_new = df[tt]
    filtered_df = df_new.dropna(subset=['Text'])
    filtered_df = filtered_df[filtered_df['Text'].str.strip() != ""]

    filtered_df['Score'] = filtered_df['Score'].astype(str)

    texts = filtered_df['Text'].values
    labels = filtered_df['Score'].values

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    num_classes = len(le.classes_)

    data = {'text': texts, 'label': labels}
    df = pd.DataFrame(data)
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, test_df

# =============================================================================
# Dataset Classes
# =============================================================================

class TextDataset(Dataset):
    """A PyTorch Dataset class for handling text data with custom tokenization and preprocessing.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing text and labels
        max_length (int): Maximum sequence length
        pad_token (int): Token ID used for padding
        vocab (dict, optional): Pre-existing vocabulary mapping
        min_freq (int): Minimum word frequency for vocabulary inclusion
    """
    def __init__(self, dataframe, max_length=128, pad_token=0, vocab=None, min_freq=2):
        self.data = dataframe
        self.max_length = max_length
        self.pad_token = pad_token
        self.min_freq = min_freq
        self.processed_data = []

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        if vocab is None:
            self.vocab, self.tokenized_texts = self._build_vocab_and_tokenize()
        else:
            self.vocab = vocab
            self.tokenized_texts = self._tokenize_texts()

        self._preprocess_data()

    def _build_vocab_and_tokenize(self):
        """Build vocabulary and tokenize texts.
        
        Returns:
            tuple: (vocabulary dict, list of tokenized texts)
        """
        word_freq = Counter()
        tokenized_texts = []

        for text in tqdm(self.data['text'], desc='lemmatizing...'):
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in self.stop_words]
            word_freq.update(tokens)
            tokenized_texts.append(tokens)

        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, freq in word_freq.items():
            if freq >= self.min_freq:
                vocab[word] = len(vocab)

        return vocab, tokenized_texts

    def _tokenize_texts(self):
        """Tokenize texts using existing vocabulary.
        
        Returns:
            list: List of tokenized texts as tensors
        """
        tokenized_texts = []

        for text in tqdm(self.data['text'], desc='tokenizing text...'):
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in self.stop_words]
            token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
            token_ids = token_ids[:self.max_length]
            tokenized_texts.append(torch.tensor(token_ids, dtype=torch.long))

        return tokenized_texts

    def _preprocess_data(self):
        """Preprocess and prepare data for model input."""
        self.processed_data = []

        for tokens, label in zip(self.tokenized_texts, self.data['label']):
            if isinstance(tokens, list):
                token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

                if len(token_ids) < self.max_length:
                    token_ids += [self.vocab['<PAD>']] * (self.max_length - len(token_ids))
                else:
                    token_ids = token_ids[:self.max_length]

                processed_text = torch.tensor(token_ids, dtype=torch.long)
            else:
                processed_text = tokens

            self.processed_data.append((processed_text, torch.tensor(label, dtype=torch.long)))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (processed text tensor, label tensor)
        """
        return self.processed_data[idx]

class TextDatasetPretrained(Dataset):
    """A PyTorch Dataset class for handling text data with pretrained tokenizers.
    
    Args:
        dataframe (pd.DataFrame): Pre-loaded dataframe with 'text' and 'label' columns.
        tokenizer (callable, optional): Transformer tokenizer (e.g., from Hugging Face).
        max_length (int): Maximum sequence length.
    """
    def __init__(self, dataframe, tokenizer=None, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_texts = self._tokenize_texts()

    def _tokenize_texts(self):
        """Tokenize texts only once, either with transformer tokenizer or NLTK."""
        tokenized_texts = []

        for text in tqdm(self.data['text'], desc='tokenizing text...'):
            if self.tokenizer is not None:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                encoding = {key: tensor.squeeze(0) for key, tensor in encoding.items()}
                tokenized_texts.append(encoding)
        return tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return self.tokenized_texts[idx], self.data['label'].values[idx]

def collate_fn(batch):
    """Custom collate function for batching text data.
    
    Args:
        batch (list): List of (text, label) tuples
        
    Returns:
        tuple: (padded_texts, labels)
    """
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return padded_texts, labels

def create_text_dataloaders(train_dir, test_dir, tokenizer=None, max_length=128, pad_token=0, 
                          batch_size=32, min_freq=2, dataset_type='imdb', 
                           cache_dir='dataset_cache'):
    """Create train and test dataloaders for text data, with caching support using descriptive file names."""
    os.makedirs(cache_dir, exist_ok=True)

    # Caching logic with descriptive file names
    if tokenizer is not None:
        tokenizer_name = tokenizer.__class__.__name__
        train_cache = os.path.join(
            cache_dir, f"{dataset_type}_train_pretrained_{tokenizer_name}_maxlen_{max_length}.pt"
        )
        test_cache = os.path.join(
            cache_dir, f"{dataset_type}_test_pretrained_{tokenizer_name}_maxlen_{max_length}.pt"
        )
        if os.path.exists(train_cache) and os.path.exists(test_cache):
            train_dataset = torch.load(train_cache, weights_only=False)
            test_dataset = torch.load(test_cache, weights_only=False)
            vocab = None
        else:
            # Only load/process raw data if cache is missing
            if dataset_type == 'imdb':
                train_df = load_imdb_data(train_dir)
                test_df = load_imdb_data(test_dir)
            elif dataset_type == 'reviews':
                train_df, test_df = load_reviews_data(train_dir)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            train_dataset = TextDatasetPretrained(train_df, tokenizer, max_length)
            test_dataset = TextDatasetPretrained(test_df, tokenizer, max_length)
            torch.save(train_dataset, train_cache)
            torch.save(test_dataset, test_cache)
            vocab = None
    else:
        min_freq_str = f"_minfreq_{min_freq}" if min_freq is not None else ""
        train_cache = os.path.join(
            cache_dir, f"{dataset_type}_train_custom{min_freq_str}_maxlen_{max_length}.pt"
        )
        test_cache = os.path.join(
            cache_dir, f"{dataset_type}_test_custom{min_freq_str}_maxlen_{max_length}.pt"
        )
        if os.path.exists(train_cache) and os.path.exists(test_cache):
            train_dataset = torch.load(train_cache, weights_only=False)
            test_dataset = torch.load(test_cache, weights_only=False)
            vocab = train_dataset.vocab if hasattr(train_dataset, 'vocab') else None
        else:
            # Only load/process raw data if cache is missing
            if dataset_type == 'imdb':
                train_df = load_imdb_data(train_dir)
                test_df = load_imdb_data(test_dir)
            elif dataset_type == 'reviews':
                train_df, test_df = load_reviews_data(train_dir)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            temp_train_dataset = TextDataset(train_df, max_length, pad_token, min_freq=min_freq)
            vocab = temp_train_dataset.vocab
            train_dataset = temp_train_dataset
            test_dataset = TextDataset(test_df, max_length, pad_token, vocab=vocab)
            torch.save(train_dataset, train_cache)
            torch.save(test_dataset, test_cache)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn if tokenizer is None else None
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn if tokenizer is None else None
    )

    return train_dataloader, test_dataloader, vocab
# =============================================================================
# Training and Evaluation
# =============================================================================

def evaluate_model(model, dataloader, criterion=None, device='cuda'):
    """Evaluate model performance on the given dataloader.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Evaluation data loader
        criterion (nn.Module, optional): Loss function. If None, CrossEntropyLoss is used.
        device (str): Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        tuple: (loss, accuracy) - Average loss and accuracy percentage
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)  # Unpack dictionary as keyword arguments
                outputs = outputs.logits  # Extract logits from SequenceClassifierOutput
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
            labels = labels.to(device).long()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def train(model, train_dataloader, val_dataloader=None, dataset_name=None, model_type=None, model_size=None, 
         num_epochs=30, lr=1e-4, optimizer_type='adam', momentum=0.9, weight_decay=0.0,
         device='cuda', save_path='results/models', show_plots=True):
    """Train a model with learning curve visualization after each epoch.
    
    Args:
        model (nn.Module or tuple): Model to train. Can be a tuple of (model, tokenizer) for pretrained transformers.
        train_dataloader (DataLoader): Training data loader
        val_dataloader (DataLoader, optional): Validation data loader
        dataset_name (str, optional): Name of the dataset (e.g., 'imdb', 'reviews')
        model_type (str, optional): Type of model architecture (e.g., 'encoder_decoder', 'encoder', 'mlp')
        model_size (str, optional): Size of the model ('small', 'medium', or 'large')
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        optimizer_type (str): Type of optimizer ('adam' or 'sgd')
        momentum (float): Momentum factor for SGD (default: 0.9)
        weight_decay (float): Weight decay (L2 penalty) (default: 0.0)
        device (str): Device to train on ('cuda' or 'cpu')
        save_path (str, optional): Base path to save the model and plots (default: 'results/training')
        show_plots (bool): Whether to show plots interactively after each epoch
        
    Returns:
        dict: Training history containing loss and accuracy metrics
    """
    # Handle case where model is a tuple (for pretrained transformers)
    if isinstance(model, tuple):
        model, tokenizer = model
    else:
        tokenizer = None
    
    # Check if model already exists
    vocab_size_str = ''
    if hasattr(model, 'embedding') and hasattr(model.embedding, 'num_embeddings'):
        vocab_size = model.embedding.num_embeddings
        vocab_size_str = f'_vocab_size_{vocab_size}'
    elif hasattr(model, 'vocab_size'):
        vocab_size = model.vocab_size
        vocab_size_str = f'_vocab_size_{vocab_size}'
    weight_decay_str = f'_wd_{weight_decay}' if weight_decay and weight_decay > 0 else ''
    model_filename = f"{dataset_name}_{model_type}_{model_size}{vocab_size_str}{weight_decay_str}_best.pth"
    model_path = os.path.join(save_path, model_filename)
    print('Model will be saved at: ', model_path)
    
    if os.path.exists(model_path):
        print(f"Warning: Found existing trained model at {model_path}")
        print("Loading existing model instead of training a new one.")
        print("If you want to train a new model, please delete the existing model file.")
        try:
            model.load_state_dict(torch.load(model_path))
            model.to(device)
        except:
            model.load_state_dict(torch.load(model_path)['net'])
            model.to(device)
        return None  # Return None to indicate model was loaded instead of trained
    
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer based on type
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        if weight_decay > 0:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Use 'adam' or 'sgd'.")
    
    model.to(device)
    
    # Create save directory if it doesn't exist
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    # Initialize history tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    # Create figure for live updates
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Determine vocab size for plot/model naming if model has embedding (text models)
    vocab_size_str = ''
    if hasattr(model, 'embedding') and hasattr(model.embedding, 'num_embeddings'):
        vocab_size = model.embedding.num_embeddings
        vocab_size_str = f'_vocab_size_{vocab_size}'
    elif hasattr(model, 'vocab_size'):
        vocab_size = model.vocab_size
        vocab_size_str = f'_vocab_size_{vocab_size}'

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_loop = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_loop:
            # Handle dictionary inputs from pretrained transformers
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)  # Unpack dictionary as keyword arguments
                outputs = outputs.logits  # Extract logits from SequenceClassifierOutput
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
            labels = labels.to(device).long()  # Convert labels to Long tensor
            
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_loop.set_postfix(loss=total_loss/len(train_loop), acc=100.*correct/total)
        
        train_loss = total_loss / len(train_dataloader)
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        if val_dataloader is not None:
            val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
        
        # Update plots
        if show_plots:
            ax1.clear()
            ax2.clear()
            
            # Plot training and validation loss
            ax1.plot(history['train_loss'], label='Train')
            if val_dataloader is not None:
                ax1.plot(history['val_loss'], label='Val')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot training and validation accuracy
            ax2.plot(history['train_acc'], label='Train')
            if val_dataloader is not None:
                ax2.plot(history['val_acc'], label='Val')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.draw()
            if save_path is not None:
                plot_filename = f"{dataset_name}_{model_type}_{model_size}{vocab_size_str}{weight_decay_str}_learning_curves.png"
                fig.savefig(os.path.join(save_path, plot_filename))
            display.clear_output(wait=True)
            display.display(fig)
            plt.pause(0.1)
            
            # Update plot title for clarity
            if show_plots:
                title = f"{dataset_name} {model_type} {model_size}"
                if vocab_size_str:
                    title += f" (Vocab size: {vocab_size})"
                fig.suptitle(title)
    
    # Save best model if validation was performed
    if val_dataloader is not None and save_path is not None:
        model_filename = f"{dataset_name}_{model_type}_{model_size}{vocab_size_str}{weight_decay_str}_best.pth"
        best_model_path = os.path.join(save_path, model_filename)
        torch.save(best_model_state, best_model_path)
        # Load the best model weights into the model
        model.load_state_dict(torch.load(best_model_path))

    plt.ioff()  # Turn off interactive mode
    return history

# =============================================================================
# Experiment Setup
# =============================================================================

def setup_experiment(config: Dict[str, Any], load_model: bool = True, model_path='') -> Tuple[Tuple[DataLoader, DataLoader], Union[nn.Module, Tuple[nn.Module, Any]]]:
    """Set up a pruning experiment with the specified configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - dataset_name (str): Name of dataset ('imdb', 'reviews', 'cifar10', 'cifar100')
            - train_dir (str): Training data directory or file path (for text datasets)
            - test_dir (str): Test data directory or file path (for text datasets)
            - text_col (str, optional): Name of text column for reviews dataset
            - label_col (str, optional): Name of label column for reviews dataset
            - model_type (str): Type of model ('encoder_decoder', 'encoder', 'mlp', 'vgg', 'resnet', 'pretrained_transformer')
            - model_size (str): Size of model ('small', 'medium', 'large', 'vgg11', 'vgg19', 'resnet18', 'resnet50')
            - batch_size (int): Batch size for dataloaders
            - max_length (int): Maximum sequence length (for text datasets)
            - min_freq (int): Minimum word frequency for vocabulary (for text datasets)
            - seed (int, optional): Random seed for reproducibility (default: 42)
            - pretrained_model_name (str, optional): Name of pretrained model for transformer models
            
    Returns:
        tuple: (train_dataloader, test_dataloader), model
            where model is either a nn.Module or a tuple of (model, tokenizer) for pretrained transformers
    """
    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    
    # Set Python random seed
    import random
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create dataloaders based on dataset type
    if config['dataset_name'] in ['cifar10', 'cifar100']:
        train_dataloader, test_dataloader = load_cifar_data(
            dataset_name=config['dataset_name'],
            batch_size=config.get('batch_size', 32)
        )
        num_classes = 10 if config['dataset_name'] == 'cifar10' else 100
    else:
        # For pretrained transformers, we need to get the tokenizer first
        tokenizer = None
        if config['model_type'] == 'pretrained_transformer':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.get('pretrained_model_name'))
        
        train_dataloader, test_dataloader, vocab = create_text_dataloaders(
            train_dir=config['train_dir'],
            test_dir=config['test_dir'],
            max_length=config.get('max_length', 128),
            batch_size=config.get('batch_size', 32),
            min_freq=config.get('min_freq', 2),
            dataset_type=config['dataset_name'],
            tokenizer=tokenizer
        )
        # Set number of classes based on dataset type
        if config['dataset_name'] == 'imdb':
            num_classes = 2  # Binary classification for IMDB
        elif config['dataset_name'] == 'reviews':
            num_classes = 5  # 5 classes for reviews dataset
        else:
            raise ValueError(f"Unknown text dataset: {config['dataset_name']}")
    
    # Build model based on type
    if config['model_type'] in ['vgg', 'resnet', 'mlp']:
        model_config = {
            'model_type': config['model_type'],
            'model_size': config['model_size'],
            'num_classes': num_classes
        }
        if config['model_type'] == 'mlp':
            model_config['input_size'] = config.get('input_size', 32*32*3)  # Default for CIFAR images
        if load_model:
            assert False #model = build_cv_model(model_config)
        else:
            model = None
    else:
        # Text models
        vocab_size = len(vocab) if vocab is not None else None
        model = build_nlp_model(
            vocab_size=vocab_size,
            num_classes=num_classes,
            model_type=config['model_type'],
            size=config['model_size'],
            pretrained_model_name=config.get('pretrained_model_name', None)
        )
    
    return (train_dataloader, test_dataloader), model 