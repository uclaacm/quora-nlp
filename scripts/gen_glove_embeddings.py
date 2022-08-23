import numpy as np
import json 
import os

# Access embeddings file: https://www.kaggle.com/datasets/takuok/glove840b300dtxt

# GOAL 1: Turn the embeddings into a json file
#         O(1) time complexity, might take a long while to load
# GOAL 2: Turn the embeddings into a text file so we can employ binary search
#         O(log(n)) time complexity

# define constants 
DATA_PATH = "../data"
EMBED_PATH = os.path.join(DATA_PATH, "embeddings/glove.840B.300d/glove.840B.300d.txt")

# open embeddings text file - not sure how long this takes file is MASSIVE. it's a CHONKER 
f = open(EMBED_PATH, "r", encoding="utf-8")

glove = {} 
for line in f.readlines(): 
    raw_line = line.split(" ") 
    word, embedding = raw_line[0], raw_line[1:]
    
    # print(f"(gen_glove_embedding_dict) > Word: {word}, Embedding: {embedding}")
    glove[word] = np.asarray(embedding, dtype='float').tolist()

f.close() 

with open(os.path.join(DATA_PATH, "bla.json"), "w") as f:
    json.dump(glove, f, indent=3)

#should probably 

