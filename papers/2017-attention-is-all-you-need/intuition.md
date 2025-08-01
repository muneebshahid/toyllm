# Transformer Architecture Intuition

## GPT vs Original Transformer

**GPT**: Uses only the **decoder** part (not encoder!) with causal masking. Goal: Predict next token given previous tokens.

**Original Transformer**: Encoder-decoder architecture. Goal: Transform one sequence into another (e.g., English → French translation).

## The Big Picture

Think of it like a translation company with two departments:

1. **Encoder**: The "understanding" department - reads and comprehends the source
2. **Decoder**: The "writing" department - generates the output using what encoder understood

## Why Each Component?

### Encoder

- **Goal**: Create rich representations of the input sequence
- **No masking needed** (usually): Can look at entire input at once to understand context
- **Self-attention**: "For each word, what other words in this sentence are relevant?"

### Decoder

- **Goal**: Generate output sequence one token at a time
- **Causal mask**: Can't cheat by looking at future tokens it's supposed to predict
- **Two attention mechanisms**:
  1. **Self-attention**: "What have I generated so far?"
  2. **Cross-attention**: "What in the source is relevant to what I'm generating now?"

## Why enc_output as K,V in cross-attention?

This is the key insight! In cross-attention:

- **Q** (queries) from decoder: "What am I looking for?"
- **K,V** (keys, values) from encoder: "Here's what I understood from the source"

Example translation: "The cat sat" → "Le chat s'est assis"

- When generating "chat", decoder asks: "I need the French word for what?"
- Encoder provides: "Here's 'cat' with all its contextual understanding"

## The Attention Mechanism Intuition

```python
# Self-attention in encoder/decoder
attention(Q=x, K=x, V=x)  # "How do parts of X relate to each other?"

# Cross-attention in decoder
attention(Q=decoder_state, K=encoder_output, V=encoder_output)
# "What parts of the source should I focus on for my current generation?"
```

## Why This Architecture?

1. **Parallel processing**: Encoder sees everything at once (fast)
2. **Conditional generation**: Decoder generates based on both source and what it already generated
3. **Alignment learning**: Cross-attention learns which source words map to which target words

For autoregressive models like GPT, we only need the decoder because we're continuing the same sequence, not transforming between different sequences.

---

# The Journey of "Hello world" → "Bonjour monde"

## Step 1: Tokenization & Embedding

```python
# Input: "Hello world" → [101, 7592, 2088] (token IDs)
# Each token gets a 512-dimensional vector
embeddings = [[0.1, -0.3, ...],  # "Hello"
              [0.5,  0.2, ...],  # "world"
              [-0.1, 0.4, ...]]  # [EOS]
```

**Intuition**: Convert words to numbers the model can work with - like giving each word a unique fingerprint.

## Step 2: Positional Encoding

```python
# Add position information
position_0 = [sin(0/10000), cos(0/10000), sin(0/10000^(2/512)), ...]
embeddings[0] += position_0  # "Hello" + "I'm first"
embeddings[1] += position_1  # "world" + "I'm second"
```

**Intuition**: Since attention has no notion of order, we add a unique "position signature" to each word. Like adding GPS coordinates to each word.

## Step 3: Encoder Processing (6 layers)

### Layer 1 - Self Attention:

```
Q: "Hello" asks: "Who should I pay attention to?"
K: All words offer: "Here's what I represent"
V: All words provide: "Here's my actual information"

Result: "Hello" learns it's a greeting, "world" is what's being greeted
```

### Layer 1 - Feed Forward:

```
Think of this as "thinking time" - each word processes what it learned
"Hello" → [greeting_vector]
"world" → [noun_being_greeted_vector]
```

**Intuition**: Each encoder layer refines understanding. Early layers catch basic patterns (grammar), later layers understand deeper meaning (semantics).

## Step 4: Encoder Output

```python
encoder_output = [
    [0.8, -0.2, ...],  # "Hello" fully understood as greeting
    [0.3,  0.9, ...],  # "world" fully understood as object
    [0.1, -0.5, ...]   # [EOS] understood as end
]
```

This is now a rich representation that captured all relationships!

## Step 5: Decoder Processing

Start with: `[BOS]` (beginning of sentence token)

### Generating "Bonjour":

1. **Self-attention**:

   - `[BOS]` asks: "What have I generated so far?"
   - Answer: "Nothing yet, I'm starting"

2. **Cross-attention**:

   - `[BOS]` asks encoder: "What should I translate first?"
   - Encoder "Hello" responds strongly: "I'm a greeting!"
   - Encoder "world" responds weakly: "I come later"

3. **Feed-forward**: Process this information

4. **Output layer**:
   ```python
   logits = [0.1, 0.2, 8.9, ...]  # 8.9 at position for "Bonjour"
   # After softmax → "Bonjour" selected
   ```

### Generating "monde":

Now we have: `[BOS] Bonjour`

1. **Self-attention** (with causal mask):

   - "Bonjour" can see `[BOS]` and itself
   - Learns: "I'm a greeting that's been generated"

2. **Cross-attention**:

   - "Bonjour" asks encoder: "What comes after a greeting?"
   - Encoder "world" now responds strongly: "Me! I'm what's being greeted"
   - Encoder "Hello" responds weakly: "Already translated"

3. **Output**: "monde" selected

## Step 6: Final Output

`[BOS] Bonjour monde [EOS]` → "Bonjour monde"

## Key Insights:

1. **Encoder**: Builds understanding by letting all words talk to each other freely
2. **Decoder**: Generates step-by-step, using both what it already generated AND the encoder's understanding
3. **Cross-attention**: The bridge - lets decoder ask "what should I translate now?"
4. **Causal mask**: Prevents cheating - decoder can't see future tokens during training

The beauty is that all these attention weights are learned! The model discovers which words should pay attention to which other words.
