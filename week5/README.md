# Retrieval Augmented Generation

There are several ways to improve the results obtained from an LLM:
- Multi-shot prompting, i.e. giving several question-answers to the model
- Use of tools
- Additional context

Retrieval Augmented Generation (RAG) is another technique that we can use to improve the results. In this we query a database, to find the most relevant info and provide it to the LLM when asking questions.

## Auto-regressive vs auto-encoding LLM

Auto-regressive LLMs take input tokens and predict a future token from the past.

Auto-encoding LLMs create the output based on the full input. These are used in several contexts, but particularly to calculate **Vector Embeddings**, representing the meaning of the input as a vector.

## Langchain

Langchain is an open-source toolset that provides frameworks interfacing many LLMs.

Pros and cons of Langchain:
- Simplifies the creation of applications using LLMs; fast time to market
- Significant adoption in the enterprise
- As APIs for LLMs have matured, converged and simplified, the need for a unifying framework has decreased. Particularly with OpenAI endpoints.
- Langchain is a more heavyweight abstraction layer than other frameworks such as LiteLLM, hence some of its code can be legacy.
