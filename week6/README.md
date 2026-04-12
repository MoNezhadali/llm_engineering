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
