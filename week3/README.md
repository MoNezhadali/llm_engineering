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
