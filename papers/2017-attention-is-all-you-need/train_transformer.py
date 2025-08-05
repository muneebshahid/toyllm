from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformer import Transformer, generate_decoder_mask, generate_padding_mask

# Simple English to French translation dataset
TRANSLATION_PAIRS = [
    ("hello", "bonjour"),
    ("world", "monde"),
    ("how are you", "comment allez vous"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("thank you", "merci"),
    ("please", "s'il vous plait"),
    ("yes", "oui"),
    ("no", "non"),
    ("cat", "chat"),
    ("dog", "chien"),
    ("house", "maison"),
    ("water", "eau"),
    ("food", "nourriture"),
    ("i love you", "je t'aime"),
    ("goodbye", "au revoir"),
    ("welcome", "bienvenue"),
    ("friend", "ami"),
    ("family", "famille"),
    ("time", "temps"),
]


class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4

    def build_vocab(self, sentences):
        word_freq = Counter()
        for sentence in sentences:
            words = sentence.lower().split()
            word_freq.update(words)

        # Add words to vocabulary
        for word, _ in word_freq.items():
            if word not in self.word2idx:
                idx = self.vocab_size
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                self.vocab_size += 1

    def encode(self, sentence, max_len=20):
        words = sentence.lower().split()
        # Add BOS and EOS tokens
        tokens = [self.word2idx["<BOS>"]]
        for word in words:
            tokens.append(self.word2idx.get(word, self.word2idx["<UNK>"]))
        tokens.append(self.word2idx["<EOS>"])

        # Pad or truncate to max_len
        if len(tokens) < max_len:
            tokens += [self.word2idx["<PAD>"]] * (max_len - len(tokens))
        else:
            tokens = tokens[: max_len - 1] + [self.word2idx["<EOS>"]]

        return tokens

    def decode(self, tokens):
        words = []
        for token in tokens:
            if token == self.word2idx["<PAD>"]:
                break
            if token == self.word2idx["<BOS>"]:
                continue
            if token == self.word2idx["<EOS>"]:
                break
            words.append(self.idx2word.get(token, "<UNK>"))
        return " ".join(words)


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_tokenizer, tgt_tokenizer, max_len=20):
        self.pairs = pairs
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src_tokens = self.src_tokenizer.encode(src_text, self.max_len)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text, self.max_len)

        return {
            "src": torch.tensor(src_tokens),
            "tgt": torch.tensor(tgt_tokens),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def create_masks(src, tgt, pad_idx=0):
    # Source mask (just padding)
    src_mask = generate_padding_mask(src, pad_idx)

    # Target mask (causal + padding)
    tgt_mask = generate_decoder_mask(tgt, pad_idx)

    return src_mask, tgt_mask


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        # Create masks
        src_mask, tgt_mask = create_masks(src, tgt)

        # Forward pass
        # Use teacher forcing: provide the entire target sequence (except last token)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]  # Shifted by one for next token prediction

        # Get model predictions
        output = model(src, tgt_input, src_mask, tgt_mask[:, :, :-1, :-1])

        # Reshape for loss calculation
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)

        # Calculate loss (ignore padding tokens)
        loss = criterion(output, tgt_output)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def generate_translation(
    model, src_tokens, src_tokenizer, tgt_tokenizer, device, max_len=20
):
    model.eval()

    # Prepare source
    src = torch.tensor([src_tokens]).to(device)
    src_mask = generate_padding_mask(src)

    # Encode source
    memory = model.encoder(src, src_mask)

    # Start with BOS token
    tgt_tokens = [tgt_tokenizer.word2idx["<BOS>"]]

    for _ in range(max_len - 1):
        # Prepare target input
        tgt = torch.tensor([tgt_tokens]).to(device)
        tgt_mask = generate_decoder_mask(tgt)

        # Get predictions
        output = model.decoder(tgt, memory, tgt_mask, src_mask)

        # Get next token (greedy decoding)
        next_token_logits = output[0, -1, :]
        next_token = next_token_logits.argmax().item()

        tgt_tokens.append(next_token)

        # Stop if EOS token is generated
        if next_token == tgt_tokenizer.word2idx["<EOS>"]:
            break

    return tgt_tokens


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create tokenizers
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()

    # Build vocabularies
    src_sentences = [pair[0] for pair in TRANSLATION_PAIRS]
    tgt_sentences = [pair[1] for pair in TRANSLATION_PAIRS]

    src_tokenizer.build_vocab(src_sentences)
    tgt_tokenizer.build_vocab(tgt_sentences)

    print(f"Source vocab size: {src_tokenizer.vocab_size}")
    print(f"Target vocab size: {tgt_tokenizer.vocab_size}")

    # Create dataset and dataloader
    dataset = TranslationDataset(TRANSLATION_PAIRS, src_tokenizer, tgt_tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Create model
    model = Transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=128,  # Smaller model for demo
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
        max_len=50,
    ).to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, device)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # Test translations
    print("\nTesting translations:")
    test_sentences = ["hello", "good morning", "cat", "i love you", "thank you"]

    model.eval()
    with torch.no_grad():
        for test_sent in test_sentences:
            # Tokenize
            src_tokens = src_tokenizer.encode(test_sent)

            # Generate translation
            tgt_tokens = generate_translation(
                model, src_tokens, src_tokenizer, tgt_tokenizer, device
            )

            # Decode
            translation = tgt_tokenizer.decode(tgt_tokens)
            print(f"{test_sent} -> {translation}")

    # Interactive translation
    print("\nEnter sentences to translate (or 'quit' to exit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == "quit":
            break

        try:
            src_tokens = src_tokenizer.encode(user_input)
            tgt_tokens = generate_translation(
                model, src_tokens, src_tokenizer, tgt_tokenizer, device
            )
            translation = tgt_tokenizer.decode(tgt_tokens)
            print(f"Translation: {translation}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
