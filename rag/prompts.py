"""Prompt engineering for relevance, grounding and citations."""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1) Rewrites follow-up questions into standalone search queries
#    ("what about its deadline?" -> "What is the deadline for certification reimbursement claims?")
CONDENSE_SYSTEM = (
    "You rewrite the user's latest question into a single standalone search query, "
    "using the chat history only to resolve pronouns and references. "
    "Keep key entities, numbers and domain terms. Do NOT answer the question. "
    "If the question is already standalone, return it unchanged. Output only the query."
)

condense_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

# 2) Grounded answer generation
ANSWER_SYSTEM = """You are a precise Document Q&A assistant.
Answer the user's question using ONLY the context below, which was retrieved from the user's uploaded documents.

Rules:
1. Ground every claim in the context. Never use outside knowledge or guess.
2. Cite sources inline using their tag, e.g. [1] or [2][3], right after the supported sentence.
3. If the context does not contain the answer, reply exactly:
   "I couldn't find this in the uploaded documents." and optionally say what related info IS available.
4. If sources conflict, point out the conflict and cite both.
5. Be concise: lead with the direct answer, then supporting details. Use bullet points for lists or steps.
6. Preserve exact numbers, dates, names and terms from the context.

Context:
{context}"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)
