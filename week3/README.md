# Notes on Week3

HuggingFace is hub for open-source models, datasets (it is the choice hub for transformers, still Keggle is for the other types of models), and Apps (including many built in Gradio)


## Diffusion vs. Transformer

Diffusion models are used for generating images. During the training images are transformed into white noise in several steps and the model learns how to create them (in backward steps). During the inference:
Text → numbers (embeddings) → guide the denoising process → image


## Sentiment Analysis

You can do sentiment analysis with transformers:

```python
from transformers import pipeline
my_simple_sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device="cuda")
result = my_simple_sentiment_analyzer("I'm super excited to be on the way to LLM mastery!")
print(result)

# Result
# [{'label': 'POSITIVE', 'score': 0.9993460774421692}]
```

## Specialized models

There are models specifically trained for specific tasks, e.g. summarization, translation, entity recognition, sentiment analysis, classification, image generation, audio generation, etc. and they are open-source, so an option is using them.

## Tokenizers

Tokenizers map between natural language and tokens (representing natural language). 

```python
# The following only works if you access to Llama3.1
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', trust_remote_code=True)
# then you can encode  a text to get the token ids, or decode the token ids to get the text + some special tokens, e.g. beginning of the prompt, ...
text = "I am excited to show Tokenizers in action to my LLM engineers"
tokens = tokenizer.encode(text)
tokens
tokenizer.decode(tokens)
```

Note that the Llama model is a base model, not a chat model, or a reasoning model. So, it does not expect user prompt, system prompts, etc. If you want to use the model similar to the chat model, you can do:

```python
# Load the instruct variant
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]
# Use the apply_chat_template method to convert messages to what the chat variant expects.
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
```

Different models, e.g. Phi-4 from Microsoft, Deepseek-V3.1, ... and their corresponding tokenizers have different structures after you apply `apply_chat_template`.

## Quantization

Quantization is a technique that reduces the numerical precision of model parameters and computations—typically from 32-bit floating point (FP32) to lower formats like 16-bit (FP16), 8-bit (INT8), or even 4-bit. This saves memory and speeds up computation, at the cost of a little accuracy. It is used for:
- Faster inference (especially on CPUs and edge devices)
- Lower memory usage (models become much smaller)
- Lower power consumption

It can happen on the neural network weights, activations (during inference time), and less often on biases since they accumulate errors stronger.

## Internals of a Pytorch LLM

If you load `Llama3.1-instruct` from huggingface and print its internals, you'll get something like:

```text
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
```

It starts with `embedding` which projects the tokens from around 128000 dimensions to 4096, then 32 layers of Neural networks, and then `lm_head` which transforms the outputs from the last layer of neural nets and specifies the probability of all the different tokens (around 128000 tokens) to be the next token.
