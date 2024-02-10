# %%
from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings 
import os
from dotenv import load_dotenv

load_dotenv()

embeddings_model = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY")) 

# %%

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
print(url[::2], key[::2])
supabase: Client = create_client(url, key)

# %%
import spacy
from langchain.text_splitter import SpacyTextSplitter
import json

file = 'TheGivingTree.txt'

with open (file, "r") as myfile:
    giving_tree_text=myfile.read()

num_questions = 10
chunk_length = len(giving_tree_text) // num_questions
print(chunk_length)

spacy_text_splitter = SpacyTextSplitter(chunk_size=chunk_length)    

spacy_segments = spacy_text_splitter.split_text(giving_tree_text)

print(len(spacy_segments))

# %%
import os.path
import pickle

embeddings_file = 'giving_tree_embeddings.pkl'

if os.path.isfile(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
else:
    embeddings = embeddings_model.embed_documents(spacy_segments)
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)

print(len(embeddings), len(embeddings[0]), embeddings[0])

#%%
j = 0
for content in spacy_segments:
    # Assuming `generate_embedding` is a function that converts your string to an embedding
    embedding = embeddings[j]
    data = {
        "source": "some website",
        "source_id": "The Giving Tree text only",
        "content": content,
        "document_id": "Giving Tree doc " + str(j),
        "author": "Shel Silverstein",
        "url": "https://en.wikipedia.org/wiki/The_Giving_Tree",
        "embedding": embedding  # Make sure this is in the format Supabase expects
    }
    supabase.table("documents").insert(data).execute()
# %%
