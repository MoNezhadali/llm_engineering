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

## Rules of thumb for RAG

- You can use several functionalities of **Langchain**, e.g. MarkdownSplitter if your documents are Markdown, but it does not necessarily work better than just chunk size plus overlap, as it did not in the experiment. The reason can be that if you use headers for split, your chunks can be too large.
- Running Evals shows that simply changing a better encoder can improve the results. One way to see what one wins/loses is to use several expensive models to see the best it gets and then optimize the objective functions, e.g. based on Week 5, day 4.

### Techniques for improving RAG

- Chunking R&D: experiment with chunking strategy
- Encoder R&D: select the best encoder model based on a test set
- Improve Prompts: genral content, the current date, relevant context and history
- Document pre-processing: use an LLM to make the chunks and/or text for encoding. You can also do semantic chunking (maybe part of chunking R&D). You can also send a table to an LLM and tell it create meaningful chunks so that I can use it later as knowledge base.
- Query rewriting: Use an LLM to convert the user's question to a RAG query.
- Query expansion: use an LLM to turn the question into multiple RAG queries.
- Re-ranking: use an LLM to sub-select from RAG results. You have obtained `n*k` prompts, you can ask an LLM to pick the most relevant ones. Specially if the chunks are going back and forth several times as context for the LLM answering the quesitons, it can help a lot with removing the unnecessary stuff and avoiding comtamination.
- Hiearchical: use an LLM to summarize at multiple levels, then when you do RAG lookup do the lookup across the summaries first (to get the coarse-grain information) and then drill down to chunks coming down from fine-grain information
- Graph RAG: retrieve content closely related to similar documents. It needs specific databases and works well in situations when your data has lots of relations between it. More often than not you can handle this by using metadata propersly.
- Agentic RAG: use agents for retrieval, combining with memory and tools such as SQL

### Semantic chunking

When you want to do semantic chunking, you can create the required functions as in Week 5, day 5, and then use LiteLLM `completion` with `response_format=Chunks`, where `Chunks` is a pydantic model:

```python
class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query")
    summary: str = Field(description="A few sentences summarizing the content of this chunk to answer common questions")
    original_text: str = Field(description="The original text of this chunk from the provided document, exactly as is, not changed in any way")

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,metadata=metadata)


class Chunks(BaseModel):
    chunks: list[Chunk]


# And later you define the completions
response = completion(model=MODEL, messages=messages, response_format=Chunks)
# NOTE: Chunks is list of Chunk, it means LLMs can handle rather complicated formats
```

#### Some python note

You can use `tqdm` to get a progress bar, while running over a list:

```python
def create_chunks(documents):
    chunks = []
    # NOTE: Here we use tqdm to get progress bar while running the code.
    for doc in tqdm(documents):
        chunks.extend(process_document(doc))
    return chunks
```

### Reranking

You can ask an LLM to rerank the chunks received:

```python
class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    print(order)
    return [chunks[i - 1] for i in order]
```

### Improving system prompt

Apparently AI performs better when it is made known to it that it's results are evaluted against certian metrics; hence it makes sense to make that known in the system prompt.

```python
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}
With this context, please answer the user's question. Be accurate, relevant and complete.
"""

# NOTE: it uses context without fstring and later uses .format method
system_prompt = SYSTEM_PROMPT.format(context=context)
```

### Querey rewriting

```python
def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention the company name unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content

# NOTE: At the end you add that IMPORTANT note to avoid the LLM to come up with more bullshit.
# NOTE: a sentence is added to the rewrite_querey method which says don't mention company name
# This shows how hacky rewriting can become worse, sometimes!
# NOTE: One possible improvement is to do query expansion, e.g. sending both rewritten and
# original question which comes next!
```

### Query expansion

As an example of this you can send both original and rewriten qestions to the LLM

```python
def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged

def fetch_context(original_question):
    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    # NOTE: there is of course a better pythonic way of doing this merge!
    chunks = merge_chunks(chunks1, chunks2)
    # NOTE: we rerank based on the original question
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]
```

#### Note on tenacity

It is a good utility that you can use for example for retrying after some backoff if you get exceptions. That can happen when you get **rate limit error**.

```python
from tenacity import retry, wait_exponential

wait = wait_exponential(multiplier=1, min=10, max=240)

@retry(wait=wait)
def process_document(document):
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]
```

#### Note on multi-processing

You can also use multi-processing to improve the process speed, but maybe you get **rate limit error**:

```python
def create_chunks(documents):
    """
    Create chunks using a number of workers in parallel.
    If you get a rate limit error, set the WORKERS to 1.
    """
    chunks = []
    # NOTE: you can also use `concurrent` or `async`
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks
```
