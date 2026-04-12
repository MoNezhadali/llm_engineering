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
