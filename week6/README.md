# Notes of Week 6

We have covered many ways to improve responses from an LLM:
- Multi-shot prompting: giving examples  in the input prompt such that the LLM knows how and what to respond.
- Tool/Function calling
- RAG/Knowledge base

All of the above are **inference time** tricks. Now we'll cover techniques for improving the model during **traning time**.

## Training

It involves setting the parameters of a model based on inputs and outputs.

### Generalization

The ability of a model to make good predictions on unseen data is called **generalization**.
LLMs are remarkably good at generalization.

### Parameters

Traditional data science models had between 20 and 200 parameters. LLMs have normally multi-billion parameters. Hence it takes a lot of time and resources to train them. Hence, you take a pre-trained model and **fine-tune** it with your business data.

### Finding datasets

The major websites/hubs for finding data are **Kaggle** and **Hugging Face**, or you can create synthetic data.

### Curating dataset

You can create a pyantic class for your data and populate it with your data making sure it follows certain standards, e.g. in the Amazon items database it can be length of description, weight, price, etc. You can also resample your data with some weights such that you get more expensive items than cheap ones of to be less biased towards items from certain categories.

You should check correlations between different features of your data.

### Preprocessing

The details you can find in Week 6, Day 2.

A good idea to do pre-processing of one's data with LLMs is to use batch mode. It is handier, as you submit and collect the results later, and you also pay less, e.g. around half price.

As for the request of outputs, it is of course best to get it in JSON, but it is more expensive!

```python
# NOTE: You create a JSONL file and then submit your request to your LLM of choice
# First you create your file by uploading your jsonl file:
with open("jsonl/0_1000.jsonl", "rb") as f:
    file = groq.files.create(file=f, purpose="batch")
# Create a batch request
# NOTE: By setting completion window to 24 hours you get half price.
response = groq.batches.create(completion_window="24h", endpoint="/v1/chat/completions", input_file_id=file.id)
# To know if the file is ready and get the id:
result = groq.batches.retrieve(response.id)
# Then you can download it
response = groq.files.content(result.output_file_id)
# And save it
response.write_to_file("jsonl/batch_results.jsonl")
```

### Traditional Machine Learning

One way to find features from Amazon products is to use `CountVectorizer` from `sklean`. Based on its descriptoin it converts a collection of text documents to a matrix of token counts. Then you can feed the description of product to it, and it can find adjectives like `luxary` or brands like `Jeep`, `Mercedes`, which can be fed to linear regression model for price prediction.

```python
documents = [item.summary for item in train]
np.random.seed(42)  # Do the nerdy 42 seed.
vectorizer = CountVectorizer(max_features=2000, stop_words='english')  # Stop words are and, or, ...
X = vectorizer.fit_transform(documents)
selected_words = vectorizer.get_feature_names_out()  # you remember get_feature_names_out?
# Use linear regression
regressor = LinearRegression()
regressor.fit(X, prices)
# Use RandomForest
subset = 15_000
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
rf_model.fit(X[:subset], prices[:subset])
# Or XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=1000, random_state=42, n_jobs=4, learning_rate=0.1)
xgb_model.fit(X, prices)
# NOTE: details are worith looking at specially the style of evaluaiton in Week 6, Day 3!
```

On Day 4, we discuss ANNs.

```python
vectorizer = HashingVectorizer(n_features=5000, stop_words='english', binary=True)
X = vectorizer.fit_transform(documents)
# NOTE: HashingVectorizer is the same as CountVectorizer just it hashes instead of keeping a dict
# and is apparently faster; however, it does not convince me that the above alone can make it faster.
# NOTE: binary=True sets it as a One-Hot-Encoder just telling existance without keeping the count.

# So you define a Neural Net
# Define the neural network - here is Pytorch code to create a 8 layer neural network

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 64)
        self.layer6 = nn.Linear(64, 64)
        self.layer7 = nn.Linear(64, 64)
        self.layer8 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output1 = self.relu(self.layer1(x))
        output2 = self.relu(self.layer2(output1))
        output3 = self.relu(self.layer3(output2))
        output4 = self.relu(self.layer4(output3))
        output5 = self.relu(self.layer5(output4))
        output6 = self.relu(self.layer6(output5))
        output7 = self.relu(self.layer7(output6))
        output8 = self.layer8(output7)
        return output8

# Training with the neural net is just a few lines of code, you can find it on Week 6, Day 4!
```

### Fine-tuning a frontier (closed source) LLM

The way fine tuning frontier models work is that they let you optimize a low-rank sub-space of the parameters (not all of them).

When you want to fine-tune a frontier model, the provider offers several different algorithms. In the case of OpenAI, examples are "Supervised fine-tuning" (the most common), "Direct preference optimization", and "reinforcement fine-tuning".

Important: Even OpenAI suggests that when you want to do fine-tuning, start with a small batch, these models are already trained on huge amounts of data!

```python
# NOTE: By defining this, you tell when I tell you this, you tell me the price!
def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [
        {"role": "user", "content": message},
        {"role": "assistant", "content": f"${item.price:.2f}"}
    ]

# NOTE: You upload the JSONL file with the purpose of fine-tune!
with open("jsonl/fine_tune_train.jsonl", "rb") as f:
    train_file = openai.files.create(file=f, purpose="fine-tune")
# Also the validation set
with open("jsonl/fine_tune_validation.jsonl", "rb") as f:
    validation_file = openai.files.create(file=f, purpose="fine-tune")
# Then you call the OpenAI fine-tuning API
openai.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=validation_file.id,
    model="gpt-4.1-nano-2025-04-14",
    seed=42,
    hyperparameters={"n_epochs": 1, "batch_size": 1},
    suffix="pricer"
)
# Then you can list events for the job id in fine-tuning to see what happens
openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10).data
# When fine-tuning is done, you can get the model name:
fine_tuned_model_name = openai.fine_tuning.jobs.retrieve(job_id).fine_tuned_model
# And then run completions with your new model
def gpt_4__1_nano_fine_tuned(item):
    response = openai.chat.completions.create(
        model=fine_tuned_model_name,
        messages=test_messages_for(item),
        max_tokens=7
    )
    return response.choices[0].message.content
```

#### Important note on fine-tuning

When you check the curves from OpenAI (the results while fine-tuning iterations) for the example of estimating prices from Amazon, you notice that at the begining the loss drops significantly (which is probably due to getting the right format of the response, i.e. just giving the estimate and the dollar sign). But later, it does not improve much, meaning it does not really learn much!

### Important conclusion

Fine-tuning frontier models is not for this purpose!

The objectives you can achieve are:
- Setting style or tone in a way that can't be achieved with prompting
- Improving the reliability of producing a type of output
- Correcting failures to follow complex prompts
- Hndling edge cases
- Performing a new skill or task that's hard to articulate in a prompt

### Deep Neural Network

As you can find in `redemption_run` script, you can build a deep neural network from scratch and beat almost all the frontier models, with a simple architecture!
