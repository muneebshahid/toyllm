import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformer import Transformer, generate_decoder_mask, generate_padding_mask
from transformers import AutoTokenizer

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

        # Tokenize with Hugging Face tokenizer
        src_encoded = self.src_tokenizer(
            src_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt_encoded = self.tgt_tokenizer(
            tgt_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "src": src_encoded["input_ids"].squeeze(0),
            "tgt": tgt_encoded["input_ids"].squeeze(0),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def create_masks(src, tgt, src_pad_idx, tgt_pad_idx):
    # Source mask (just padding)
    src_mask = generate_padding_mask(src, src_pad_idx)

    # Target mask (causal + padding)
    tgt_mask = generate_decoder_mask(tgt, tgt_pad_idx)

    return src_mask, tgt_mask


def train_epoch(
    model, dataloader, optimizer, criterion, device, src_tokenizer, tgt_tokenizer
):
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        # Create masks
        src_mask, tgt_mask = create_masks(
            src, tgt, src_tokenizer.pad_token_id, tgt_tokenizer.pad_token_id
        )

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
    model, src_text, src_tokenizer, tgt_tokenizer, device, max_len=20
):
    model.eval()

    # Tokenize source
    src_encoded = src_tokenizer(
        src_text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    src = src_encoded["input_ids"].to(device)
    src_mask = generate_padding_mask(src, src_tokenizer.pad_token_id)

    # Encode source
    memory = model.encoder(src, src_mask)

    # Start with BOS token (using tokenizer's cls_token_id as BOS)
    tgt_tokens = [tgt_tokenizer.cls_token_id]

    for _ in range(max_len - 1):
        # Prepare target input
        tgt = torch.tensor([tgt_tokens]).to(device)
        tgt_mask = generate_decoder_mask(tgt, tgt_tokenizer.pad_token_id)

        # Get predictions
        output = model.decoder(tgt, memory, tgt_mask, src_mask)

        # Get next token (greedy decoding)
        next_token_logits = output[0, -1, :]
        next_token = next_token_logits.argmax().item()

        tgt_tokens.append(next_token)

        # Stop if EOS token is generated (using sep_token_id as EOS)
        if next_token == tgt_tokenizer.sep_token_id:
            break

    return tgt_tokenizer.decode(tgt_tokens, skip_special_tokens=True)


def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create tokenizers - using a small pre-trained model for demonstration
    # You could also train your own tokenizer with Hugging Face's tokenizers library
    src_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tgt_tokenizer = AutoTokenizer.from_pretrained("camembert-base")  # French tokenizer

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
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_tokenizer.pad_token_id
    )  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    print("\nTraining...")
    for epoch in range(num_epochs):
        loss = train_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            device,
            src_tokenizer,
            tgt_tokenizer,
        )

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # Test translations
    print("\nTesting translations:")
    print("-" * 50)
    test_sentences = [
        "hello",
        "good morning",
        "cat",
        "dog",
        "i love you",
        "thank you",
        "water",
        "goodbye",
    ]

    model.eval()
    with torch.no_grad():
        for test_sent in test_sentences:
            # Generate translation
            translation = generate_translation(
                model, test_sent, src_tokenizer, tgt_tokenizer, device
            )
            print(f"{test_sent:15} → {translation}")

    # Show attention weights for one example
    print("\n" + "=" * 50)
    print("Analyzing 'hello' translation:")
    print("=" * 50)

    # Get attention weights
    src_encoded = src_tokenizer(
        "hello",
        return_tensors="pt",
        padding="max_length",
        max_length=20,
        truncation=True,
    )
    src = src_encoded["input_ids"].to(device)
    src_mask = generate_padding_mask(src, src_tokenizer.pad_token_id)

    # Get encoder output
    with torch.no_grad():
        enc_output = model.encoder(src, src_mask)
        print(f"Encoder output shape: {enc_output.shape}")

        # Generate step by step to see attention
        tgt_tokens = [tgt_tokenizer.cls_token_id]
        tgt = torch.tensor([tgt_tokens]).to(device)
        tgt_mask = generate_decoder_mask(tgt, tgt_tokenizer.pad_token_id)

        # Get decoder output for first step
        # We would need to modify the model to return attention weights
        # For now, just show the generation process
        output = model.decoder(tgt, enc_output, tgt_mask, src_mask)
        probs = torch.softmax(output[0, -1, :], dim=0)
        top5_probs, top5_idx = probs.topk(5)

        print("\nTop 5 predictions after <BOS>:")
        for i in range(5):
            word = tgt_tokenizer.decode([top5_idx[i].item()])
            prob = top5_probs[i].item()
            print(f"  {word:15} {prob:.3f}")


if __name__ == "__main__":
    main()
