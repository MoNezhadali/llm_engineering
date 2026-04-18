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

## Five important hyper-parameters for QLoRA

- Target modules: e.g. attention layers, or adding other layers
- rank (`r`): number of dimensions of the LoRA matrices
- Alpha: After multiplying `lora_a` and `lora_b` how much you scale it before adding it to the base model. It's normally `2*r`
- Quantization: How much you reduce the precision of the base model
- Dropout: randomly droping out a some of the caluculations in the neurons (e.g. `10` or `20` percent) to make sure the neural net generalizes well and does not overfit the data. Doing so, if the neural network wants to perform well, it has to adjust its parameters everywhere, not only in a very specific area.

## Five important hyper-parameters for Training

- Epochs: how many complete passes the learning algorithm makes over the full training dataset. It makes sence to have epochs greater than zero since at every step you learn a little bit, and you have batches, so it moves in the collective direction (not all the info in one specific input is learned)
- Batch size: how many of these data points are put in a training batch, rule of thumb is to fit as big batch size as it can fit in your GPU (normally powers of `2`, lol), but remember the larger your batch size, the more approximate your correction will be (corrections may cancel out each other), but it is not necessarily bad.
- Learning rate: how big of a step you take in each step of training. You start with `0.0001`, this one is noramlly a power of `10`. There are some **learning rate schedulers** which can adjust your learning rate based on where you are.
- Gradient Accumulation: in the back propagate, you compute gradients, you can process multiple smaller batches (micro-batches), accumulate (sum) gradients across them, only update weights after several steps.
- Optimzer: you can do the simplest option (stochastic gradient decent, SGD), or the most popular (Adaptive Moment Estimation, Adam)

## Weight and Biases (`wandb`)

`wandb` is a free ML tool for tracking, visualizing, and managing experiments

## Hyper parameters for the experiment

```python
# Constants

BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "ed-donner" # your HF name here!

LITE_MODE = True

DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

RUN_NAME =  f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
if LITE_MODE:
  RUN_NAME += "-lite"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Hyper-parameters - overall

EPOCHS = 1 if LITE_MODE else 3
BATCH_SIZE = 32 if LITE_MODE else 256
MAX_SEQUENCE_LENGTH = 128
GRADIENT_ACCUMULATION_STEPS = 1

# Hyper-parameters - QLoRA

QUANT_4_BIT = True
LORA_R = 32 if LITE_MODE else 256
LORA_ALPHA = LORA_R * 2
ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]
TARGET_MODULES = ATTENTION_LAYERS if LITE_MODE else ATTENTION_LAYERS + MLP_LAYERS
LORA_DROPOUT = 0.1

# Hyper-parameters - training

LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.01
LR_SCHEDULER_TYPE = 'cosine'
WEIGHT_DECAY = 0.001
OPTIMIZER = "paged_adamw_32bit"

capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8

# Tracking

VAL_SIZE = 500 if LITE_MODE else 1000
LOG_STEPS = 5 if LITE_MODE else 10
SAVE_STEPS = 100 if LITE_MODE else 200
LOG_TO_WANDB = True
```

Then you can run your experiment:

```python
# pick the right quantization
quant_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
  bnb_4bit_quant_type="nf4"
)
# Load the Tokenizer and the Model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id
# LoRA Parameters
lora_parameters = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)
# Training parameters
train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    logging_steps=LOG_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=not use_bf16,
    bf16=use_bf16,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to="wandb" if LOG_TO_WANDB else None,
    run_name=RUN_NAME,
    max_length=MAX_SEQUENCE_LENGTH,
    save_strategy="steps",
    hub_strategy="every_save",
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS
)
# Define fine-tuning experiment
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train,
    eval_dataset=val,
    peft_config=lora_parameters,
    args=train_parameters
)
# NOTE: This SFT trainer does all the training and reports to weights and biases platform
```

### Stopping a run in Colab

If you stop a run in Colab, still there will be saved in HuggingFace. So instead of running over 800K data points, you can run is for 100K and see the results! In general makes sense, that one saves the results regularly during training, so see the progress live, and also be felxible with stops if it makes sense to do so.

### Some notes on training

- The training curve looks smoother when there are larger batch sizes.
- If you set up large epochs, the training scheduler applies to the entire training, not each epoch!
- When there are several epochs, you can see in the training loss curve that there are several jumps, happening because the model has seen the exact same data before and been trained by it. However, if there is a huge jump that's not good news, maybe sign of overfitting!
- Note that HuggingFace is just a git repository! Meaning that you can go to HuggingFace and browse `date_of_the_day_hyper_parameters_or_anything` and pick the run (commit) at `6200`th step!
- NOTE: the loss that was used for this experiment was not any measure of the difference between the actual price and prediction, but `-log(P(true value))`! This function also works very well with **BackProp** algorithm for calculating derivatives! That why it is a good choice! It is called **cross-entropy loss**! It works perfectly fine with classification problems including this one!

### Cross-entropy loss

This is cross-entropy loss:
```python
loss = - log(P(true_value))
```

## The final fin-tuned model

It is `2.2 GB` for the Llama quantized model and around `1.5 GB` for all the fine-tuning parameters affecting both `attention` layers, and `mlp` layers! And it performed better than all models, including the best frontier models!

![Final Result](final_result.png)
