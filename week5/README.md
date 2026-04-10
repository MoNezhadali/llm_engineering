# Retrieval Augmented Generation

There are several ways to improve the results obtained from an LLM:
- Multi-shot prompting, i.e. giving several question-answers to the model
- Use of tools
- Additional context

Retrieval Augmented Generation (RAG) is another technique that we can use to improve the results. In this we query a database, to find the most relevant info and provide it to the LLM when asking questions.

This simple approach; however, can also cause problems, e.g. change of context while asking consecutive questions may confuse the LLM, chunks not getting the full context. Hence very often solutions to RAG are very hacky.

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

### Some functionalities of Langchain

`RecursiveCharacterTextSplitter` can be used to create chunks of a document in a hierarchy. In this you can also define `chunck_overlap` argument to make sure chunks are not created by cutting some paragraph into two pieces which detoriorates the meaning in the paragraphs.

We can use `Chroma` object from `Langchain` to create a Chroma database. Chroma is a free open-source database that has SQLite running behind it. Using `from_documents` method and giving it the chunks, embedding model, and database address, you can create and populate your vector database in one line.

You can use `TSNE` to visualize the embeddings in lower spaces, e.g. `2D`.

You can use `Chroma` class of `langchain_chroma` to create a fully fledged auto-encoder model:

```python
from langchain_chroma import Chroma
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
retriever.invoke("Who is Avery?")
# This provides the corresponding documents with their IDs, metadaa, source (which document they come from originally) and page content.
```

Then you can combine it with invokation of the LLM, briefly explained in day 3, and create a basic RAG. A better version of that is available at [implementation](./implementation/).

## Vector databases

There are many options for that; e.g. open-source: Chroma, Qdrant, ..., paid subscriptions: weaviate, pinecone, ... However, since most of the database systems (Postgres, Mango, Elastic, ...) are now supporting vector databases natively, going for paid enterprise solutions is less relevant.

## Evals

Evalutions are a scientific technique which can help with reassuring that the RAG workflow is working fine. In order to create Evals, you should have the followings ready:
- An example set of questions with the correct answers identified and reference answers provided.
- Measure retrieval: MRR (Mean Reciprocal Rank), nDCG (normalized Discounted Cumulative Gain), Recall@K, and Precision@K
- Measure Answers: you can use LLM-as-a-judge to score provided answers against criteria like accuracy, completeness, and relevance.

### MRR

The following is the formula for **MRR**: the average rank (among chunks you retrieve) at which you can find the correct answer

```
MRR = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{{rank}_i}
```

### nDCG

It looks at the distribution of the chunks and measures how likely it is to get the most relevant ones at the first chunks

### Recall@k

Proportion of tests where the relevant context was in the top `k` chunks, or if you have multiple keywords to look for, **keyword coverage** is a similar recall metric, what proportion of your keywords were in the first `k` chunks

### Precision@k

Proportion of the top `k` chunks that are relevant.

## JSONL

The `.jsonl` is a type of file whose every line is a valid JSON object, but the entire file is not. It is very popular for test data, since you can add text examples one by one to the file. It is the JSON version of a `.csv` file.

## LLM as a judge

In order to use LLM-as-a-judge in a testing workflow, if you are using `LiteLLM` you can also ask the response to be in a specific format, defined by `Pydantic`:

```python
class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )

judge_response = completion(model=MODEL, messages=judge_messages, response_format=AnswerEval)
```
