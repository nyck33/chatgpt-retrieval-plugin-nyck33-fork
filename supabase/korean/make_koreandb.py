# %%
# https://platform.openai.com/docs/guides/embeddings/use-cases
from supabase import create_client, Client
#from langchain_openai import OpenAIEmbeddings 
from openai import OpenAI 
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
print(OPENAI_API_KEY[::2])
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
print(EMBEDDING_MODEL)
EMBEDDING_DIMENSION = os.environ.get("EMBEDDING_DIMENSION")
print(EMBEDDING_DIMENSION)

#embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL, deployment=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSION) 

#print(f'embeddings_model: {embeddings_model}')
client = OpenAI(api_key=OPENAI_API_KEY)
# %%
# get supabase env vars and make client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
# https://lhbeoisvtsilsquybifs.supabase.co
# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxoYmVvaXN2dHNpbHNxdXliaWZzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcwNzU0MzkwMCwiZXhwIjoyMDIzMTE5OTAwfQ.vRXYacgp-e6HYVtXv-fekRTA7r4y8k0SjjtMB5LYqF8
print(url, key)
supabase: Client = create_client(url, key)

# %%
# split the text into chunks
import spacy
from langchain.text_splitter import SpacyTextSplitter
import json

file = 'TheGivingTree.txt'
spacy_segments = []
if not os.path.isfile('embeddings.csv'):
    with open(file, "r") as myfile:
        giving_tree_text=myfile.read()

    num_questions = 10
    chunk_length = len(giving_tree_text) // num_questions
    print(chunk_length)

    spacy_text_splitter = SpacyTextSplitter(chunk_size=chunk_length)    

    spacy_segments = spacy_text_splitter.split_text(giving_tree_text)

    print(len(spacy_segments))
else:
    spacy_segments = []

#%% 
# check each spacy_segment for excessive white space
print(spacy_segments[0])
# each spacy segment should not have more than one space between words or punctuation and words
for i in range(len(spacy_segments)):
    spacy_segments[i] = ' '.join(spacy_segments[i].split())
    print(spacy_segments[i], "\n")

#%%
# make dataframe of three columns: segment_num, segment_text, embedding
import pandas as pd

import os.path

loaded_from_file = False

if os.path.isfile('embeddings.csv'):
    df = pd.read_csv('embeddings.csv', index_col='segment_num')
    spacy_segments = df['segment_text'].tolist()
    loaded_from_file = True
else:
    # Assuming spacy_segments is a list of text segments
    df = pd.DataFrame({
        'segment_text': spacy_segments
    })
    df['embedding'] = None  # Placeholder for the actual embeddings you will generate

    # Set 'segment_num' as index
    df.index = range(len(spacy_segments))
    df.index.name = 'segment_num'

df.head()

# %%
# get embeddings
def get_embeddings(text, model=EMBEDDING_MODEL):
    text = text.replace('\n', ' ')
    response = client.embeddings.create(input=[text], model=model, dimensions=int(EMBEDDING_DIMENSION))
    embeddings = response.data[0].embedding
    return embeddings

# %%
embeddings = []
if not loaded_from_file:
    for i, row in df.iterrows():
        text = row['segment_text']
        embedding = get_embeddings(text)
        embeddings.append(embedding)
        df.at[i, 'embedding'] = embedding
    df.to_csv('embeddings.csv')
else:
    embeddings = df['embedding'].tolist()
'''
import os.path
import pickle

load_from_file = False

embeddings_file = 'giving_tree_embeddings.pkl'

if os.path.isfile(embeddings_file) and load_from_file:
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
else:
    embeddings = embeddings_model.embed_documents(spacy_segments)
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)

print(len(embeddings), len(embeddings[0]), embeddings[0])
'''
# %%
# check df text column for excessive white space
print(df['segment_text'][0])
# for 
#%%
# look at embeddings
print(len(embeddings), len(embeddings[0]), type(embeddings[0]))
print(embeddings[0])

# %% check df's embedding column for type, shape, etc
print(df['embedding'].dtype)
print(df['embedding'].shape)
print(type(df['embedding'][0]))
print(len(df['embedding'][0]))
#%%
j = 0
for content in spacy_segments:
    # Assuming `generate_embedding` is a function that converts your string to an embedding
    embedding = embeddings[j]
    data = {
        "source": "file", #email, chat or file enum
        "source_id": "The Giving Tree text only",
        "content": content,
        "document_id": "Giving Tree doc x",
        "author": "Shel Silverstein",
        "url": "https://en.wikipedia.org/wiki/The_Giving_Tree",
        "embedding": embedding  # Make sure this is in the format Supabase expects
    }
    supabase.table("documents").insert(data).execute()
    j += 1
# %%
