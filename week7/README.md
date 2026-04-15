# Notes of Week 7

## How LoRA works

In a neural network, it selects some target modules, e.g. only a few layers which are more effective, and then uses some low-rank adaptors to adapt them to the new data. For example Llama is too big to train, we train something smaller and add that to Llama.

## QLoRA

Even `3 billion` variables is too large; `3 billion * 32 bits = 13 GB`.
One can keep the number of weights but reduce their precision. Model performance deteriorates but the impact is surprisingly small! one can reduce `32` bits to `8` bits or even `4` bits!

Some notes:
- The `4 bits` are interpreted as floats not ints (They choose them as normally distributed floats, such that the distribution of the final numbers is similar to the original values, and the model does not lose too much information)
- The adaptor matrices are still in 32 bits (they have fewer number of parameters and we can afford it!). So Q (for quantizatoin) applies to the base matrix, not to the Low rank matrix!

You can use `bitsandbytes` library to load quantized models:

```python
# NOTE: Quantized 8-bit Llama model is loaded like this
BASE_MODEL = "meta-llama/Llama-3.2-3B"
quant_config = BitsAndBytesConfig(load_in_8bit=True)
# Then load the model with quant_config
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

# NOTE: Quantized 4-bit Llama with double-quant which even saved more memory
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4")
# This part is the same:
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

# You can print out the memory footprint of the model by
print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.2f} GB")
# Which prints out 2.2 GB
# If you load a fine-tuned model (base_model + the low-rank matrix):
fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e9:,.2f} GB")
# It prints out 2.27 GB which means the low-rank matrix is only 70 MB!
```

Following is the architecture of a **base model**:

```text
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 3072)
    (layers): ModuleList(
      (0-27): 28 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear4bit(in_features=3072, out_features=3072, bias=False)
          (k_proj): Linear4bit(in_features=3072, out_features=1024, bias=False)
          (v_proj): Linear4bit(in_features=3072, out_features=1024, bias=False)
          (o_proj): Linear4bit(in_features=3072, out_features=3072, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear4bit(in_features=3072, out_features=8192, bias=False)
          (up_proj): Linear4bit(in_features=3072, out_features=8192, bias=False)
          (down_proj): Linear4bit(in_features=8192, out_features=3072, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((3072,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=3072, out_features=128256, bias=False)
)
```

Which can be compared to the architecture of a **fine-tuned model** with QLoRA:

```text
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(128256, 3072)
        (layers): ModuleList(
          (0-27): 28 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=3072, out_features=3072, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3072, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=3072, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (k_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=3072, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3072, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (v_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=3072, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3072, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (o_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=3072, out_features=3072, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3072, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=3072, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear4bit(in_features=3072, out_features=8192, bias=False)
              (up_proj): Linear4bit(in_features=3072, out_features=8192, bias=False)
              (down_proj): Linear4bit(in_features=8192, out_features=3072, bias=False)
              (act_fn): SiLUActivation()
            )
            (input_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
            (post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
          )
        )
        (norm): LlamaRMSNorm((3072,), eps=1e-05)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (lm_head): Linear(in_features=3072, out_features=128256, bias=False)
    )
  )
)
```

As you see each of the self attention layers has `lora_A (3072 * 32)` and `lora_B (32 * 3072)`, so `lora_A * lora_B` is a low rank matrix which shrinks the `3072` dimension space into `32` dimensions and brings it back to `3072` dimensions.

## Token count cut-off

When you fun fun-tuning, you have to set up in advance what is the maximum number of tokens in any of the training data. Because it has to prepare itself for that scenario. The amount of memory in the GPU is linearly proportional to the maximum number of tokens. So you should check your fine-tuning data and set a cut-off! If you do not do it, all the shorter sequences are padded with a special token and since there will be very many of those padded character, it will drastically slow down the fine-tuning process!

```python
token_counts = [item.count_tokens(tokenizer) for item in tqdm(items)]
CUTOFF = 110
cut = len([count for count in token_counts if count > CUTOFF])
# NOTE: You normally do not get rid of the fine-tuning data, you just truncate the long ones.
```

## Some note on training for prices

When you want to train for some quantity, it is best to help the model not to get busy with the details after some point, e.g. cents do not matter but the model will try to predict them, and that triviality can confuse the model, since it is trying to get the number of cents right as much as it is trying to predict the dollars right. Remember we are asking an NLP (predictor of the next token) to solve something numeric!

Note: you should calculate this max number and give it HuggingFace before fine-tuning!

Note: All three digit-numbers are mapped to one token in lamma, unlike some other model. So it makes it suitable for predicting the price of an item under `$1000`! It basically solves a classification problem. I, personally, think it would still work if it were more (It may even be more appropriate predicting units, tens, hundreds, ...)!

## Format

If you check the `Item` class from the `pricer` module:

```python
class Item(BaseModel)
    ...
    def to_datapoint(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}
```

You see that the columns of the dataset are chosen to be `prompt` and `completion`, these are exactly what `HuggingFace` expects.

### Important Note

There are two types of LLM model:
- Base models (`meta-llama/Llama-3.2-3B`)
- Chat models (`meta-llama/Llama-3.2-3B-instruct`)
- Actually there is a third type as well: thinking or reasoning models

Base models get a chunk of text and predict the next token. They expect `prompt` and `completion`.

OpenAI came up with the idea of creating the `system` prompt, `user` prompt, `assistant` prompt, chat-style type of input and output, and as a result it became really good at answering rather than just predicting the text.

In the case of price-prediction, we could try either of the two types, but since the instructions do not really matter the problem is rather clear, we choose the base model.
