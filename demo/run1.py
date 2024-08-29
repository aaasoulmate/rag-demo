import sys, os
sys.path.append("../utils/")

from llm1 import EmbeddingModel, VectorStoreIndex, LLM

# Create embedding model
print("> Create embedding model...")
embed_model_path = '../models/AI-ModelScope/bge-small-zh-v1___5'
embed_model = EmbeddingModel(embed_model_path)

# Create index
print("> Create index...")
doecment_path = '../datasets/knowledge.txt'
index = VectorStoreIndex(doecment_path, embed_model)

# Query the index
question = '介绍一下广州大学'
print('> Question:', question)

context = index.query(question)
print('> Context:', context)

# Create LLM model
print("> Create Yuan2.0 LLM...")
model_path = '../models/IEITYuan/Yuan2-2B-Mars-hf'
# model_path = './IEITYuan/Yuan2-2B-July-hf'
llm = LLM(model_path)

# Generate without RAG
print('> Without RAG:')
llm.generate(question, [])

# Generate with RAG
print('> With RAG:')
llm.generate(question, context)
