# Notes of Week 4

## General notes about choosing models

When choosing an LLM model, all the details of that model can be found in the model card provided by the provider of that model.

Be careful about licences of both open-source and closed-source models.

The Chinchilla rule: number of parameters ~ number of training tokens. But with techniques like RAG where we infuse information during inference, or methods for distilling the models, it has become less relevant.

Here are some famouse benchmarks:
- Google Proof QA; PhD level expertice in physics, chemistry, and biology is required to answer 65 percent
- MMLU-pro; language understanding benchmark (10 choice per question)
- AIME; Math olympiad level questions
- LiveCodeBench; Coding benchmark whose problems are based on coding challenges in Leetcode, Codeforces, ...
- MuSR; for reasoning, for example reading a text and asking who has the means, motive, and opportunity
- HLE (humnanities last exam); Super-human intelligence, 2500 of the toughest, subject-diverse, multi-modal type of questions.

Another good way to compare LLMs againts each other is leaderboards. The famous ones are:
- Artificial Analysis
- Vellum
- Scale.com
- Huggingface
- LiveBench

# Some python note

You can run a string in python using the `exec` command:

```python
tmp = "print(2+2)"
exec(tmp)
# It outputs 4
```
