# Notes on Week 8

## Agents to be created

Here are the list of agents we create
- Scanner agent: An agen subscribing to the RSS feeds and identifying good bargains
- Ensemble agent: estimates prices using multiple models
- Messaging agent: sends push notification
- Planning agent: coordinates activities
- The agent framework; with memory and logging
- The UI; using Gradio

### IMPORTANT NOTE

The agents are developed in [Agents folder](./agents/). Go through them for any question you have!
Also look at `deal_agent_framework.py`, it adds persistence and memory to the agent! For the `UI` you can check `price_is_right.py`

## Modal

It is a platform for deployment of AI inference! After you create your modal API token, you set it using `uv` like this:

```bash
modal token set --token-id your-token-id --token-secret your-token-secret
```

**Modal** helps run your python code on their servers, check `hello.py`, all you need to do is define your device and what you want installed on it:

```python
from modal import Image
image = Image.debian_slim().pip_install("requests")
```

and then run `hello.remote` instead of `hello.local`!

## Deploying Llama using Modal

Just like this:

```python
import modal
from modal import Image

# Setup

app = modal.App("llama")
image = Image.debian_slim().pip_install("torch", "transformers", "accelerate")
secrets = [modal.Secret.from_name("huggingface-secret")]
GPU = "T4"
MODEL_NAME = "meta-llama/Llama-3.2-3B"

# NOTE: Here is how you define GPU, secrets, and other characteristics!
@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def generate(prompt: str) -> str:
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_new_tokens=5)
    return tokenizer.decode(outputs[0])
```

Or you can also run the model you have trained in HuggingFace!

Remember you should not just run your codes on Modal, you should deploy them!

In order to get things faster and more efficient, you can add volumes to your deployments and cache your models there! It is an example of **serverless AI**. You can of course set `MIN_Containers=1`, to make sure you always have one container up and running! Of course it is pricier!

## Object-oriented agents

You can make a parent class `Agent` and make logging and all shared characteristics there!

You can check Week 8 Day 1 for how to create a service for the specialist agent (for estimating prices)!

## Creating a Vector database

Here is how you create a vector database and add metadata to it:

```python
# NOTE: Define a free and fast encoder which can be run locally
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Check if the collection exists; if not, create it

collection_name = "products"
existing_collection_names = [collection.name for collection in client.list_collections()]

if collection_name not in existing_collection_names:
    collection = client.create_collection(collection_name)
    for i in tqdm(range(0, len(train), 1000)):
        documents = [item.summary for item in train[i: i+1000]]
        vectors = encoder.encode(documents).astype(float).tolist()
        metadatas = [{"category": item.category, "price": item.price} for item in train[i: i+1000]]
        ids = [f"doc_{j}" for j in range(i, i+1000)]
        ids = ids[:len(documents)]
        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)

collection = client.get_or_create_collection(collection_name)

# NOTE: Here is how you query it afterwards:
def find_similars(item):
    vec = vector(item)
    results = collection.query(query_embeddings=vec.astype(float).tolist(), n_results=5)
    documents = results['documents'][0][:]
    prices = [m['price'] for m in results['metadatas'][0][:]]
    return documents, prices
```

## Ensemble-based approach

When all your models are ready, you should do a linear regression on the validation set to find the best combination of weights with which the errors are minimized!

```python
# After linear regression:
def ensemble(item):
    price1 = get_price(gpt_5__1_rag(item))
    price2 = specialist(item)
    price3 = deep_neural_network(item)
    return price1 * 0.8 + price2 * 0.1 + price3 * 0.1
```

Of course, you should do it by abiding with python best practices, which are under [Agents directory](./agents/)!

## Note on structured LLM outputs

What basically OpenAI does is that when the model is generating response, it zeros out all the probabilities which result in invalid JSON objects step by step, hence the response will always be a valid JSON object!

Then instead of ``openai.chat.completion.create` you can use `openai.chat.completion.parse` method:

```python
# After defining your pydantic objects:
class Deal(BaseModel):
    """
    A class to Represent a Deal with a summary description
    """

    product_description: str = Field(
        description="Your clearly expressed summary of the product in 3-4 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a short paragraph of text for each item you choose."
    )
    price: float = Field(
        description="The actual price of this product, as advertised in the deal. Be sure to give the actual price; for example, if a deal is described as $100 off the usual $300 price, you should respond with $200"
    )
    url: str = Field(description="The URL of the deal, as provided in the input")


class DealSelection(BaseModel):
    """
    A class to Represent a list of Deals
    """

    deals: List[Deal] = Field(
        description="Your selection of the 5 deals that have the most detailed, high quality description and the most clear price. You should be confident that the price reflects the deal, that it is a good deal, with a clear description"
    )

# You can use the parse method:
response = openai.chat.completions.parse(model=MODEL, messages=messages, response_format=DealSelection, reasoning_effort="minimal")
# NOTE: the results object will be DealSelection object populated with the JSON that came back:
results = response.choices[0].message.parsed
```

## Pushing notifications to your phone

If you want to send notifications to your own phone, you can register in a platfrom like `Pushover.net` get your `PUSHOVER_USER` and `PUSHOVER_TOKEN` and then send `post` requests to them with your message and credentials!

## Hallmarks of an agentic AI solution

- Breaking a larger problem into smaller steps carried out by individual processes
- Using tools/funcion calling/structured outputs
- An agent environment in which agents can collaborate
- A planning agent that coordinates activities
- Autonomy and memory - existing beyond a chat with a human

## Use of tools

You should create tool description JSON objects:

```python
# NOTE: If you use an agent framework, it creates these JSON objects for you!
scan_function = {
        "name": "scan_the_internet_for_bargains",
        "description": "Returns top bargains scraped from the internet along with the price each item is being offered for",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    }

estimate_function = {
    "name": "estimate_true_value",
    "description": "Given the description of an item, estimate how much it is actually worth",
    "parameters": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "The description of the item to be estimated"
            },
        },
        "required": ["description"],
        "additionalProperties": False
    }
}

notify_function = {
    "name": "notify_user_of_deal",
    "description": "Send the user a push notification about the single most compelling deal; only call this one time",
    "parameters": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "The description of the item itself scraped from the internet"
            },
            "deal_price": {
                "type": "number",
                "description": "The price offered by this deal scraped from the internet"
            }
            ,
            "estimated_true_value": {
                "type": "number",
                "description": "The estimated actual value that this is worth"
            }
            ,
            "url": {
                "type": "string",
                "description": "The URL of this deal as scraped from the internet"
            }
        },
        "required": ["description", "deal_price", "estimated_true_value", "url"],
        "additionalProperties": False
    }
}
```

Afterwards, you will append all the results from tool calls to the results with `{"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id}`:

```python
def handle_tool_call(message):
    """
    Actually call the tools associated with this message
    """
    results = []
    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
    return results
```

Then you can run a loop like this to make sure the agent does what you want:

```python
system_message = "You find great deals on bargain products using your tools, and notify the user of the best bargain."
user_message = """
First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
Then pick the single most compelling deal where the price is much lower than the estimated true value, and use your tool to notify the user.
Then just reply OK to indicate success.
"""
messages = [{"role": "system", "content": system_message},{"role": "user", "content": user_message}]

# NOTE: Here we create the loop
done = False
while not done:
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        results = handle_tool_call(message)
        messages.append(message)
        messages.extend(results)
    else:
        done = True
response.choices[0].message.content
```

You can find the better implementaiton in [Autonomous Agent](./agents/autonomous_planning_agent.py)
