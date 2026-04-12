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
