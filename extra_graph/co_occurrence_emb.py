import pickle

import pandas as pd
import networkx as nx
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 或 "true"
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('german'))


class CooccurGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(CooccurGraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x


# 1. Load and preprocess text data
df = pd.read_csv("your_path", sep="\t")
reports = df['report'].dropna().tolist()
# reports = reports[0:10]  # 示例：前10条

# 2. Build co-occurrence graph
co_occurrence = defaultdict(int)
all_tokens = []

for report in reports:
    tokens = word_tokenize(report.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation and word.isalpha()]
    all_tokens.append(tokens)
    unique_tokens = list(set(tokens))
    for i in range(len(unique_tokens)):
        for j in range(i + 1, len(unique_tokens)):
            pair = tuple(sorted([unique_tokens[i], unique_tokens[j]]))
            co_occurrence[pair] += 1

# 3. Create graph with edges weighted by co-occurrence counts
G = nx.Graph()
for (w1, w2), count in co_occurrence.items():
    if count >= 2:
        G.add_edge(w1, w2, weight=count)

# 4. Create node features (using BERT embeddings)
vocab = sorted(list(G.nodes))
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {v: k for k, v in word2idx.items()}

# 5. Create edge index and edge weight for PyG
edge_index = []
edge_weight = []

for u, v, attr in G.edges(data=True):
    i, j = word2idx[u], word2idx[v]
    edge_index.append([i, j])
    edge_index.append([j, i])  # 双向图
    edge_weight.append(attr['weight'])
    edge_weight.append(attr['weight'])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight, dtype=torch.float)

# 6. Get BERT embeddings for each node (word)
tokenizer = AutoTokenizer.from_pretrained('your_path/MedCPT_Query_Encoder')
model = AutoModel.from_pretrained('your_path/MedCPT_Query_Encoder')
model.eval()


def get_bert_embedding(word):
    with torch.no_grad():
        inputs = tokenizer(word, return_tensors='pt', truncation=True, max_length=10)
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0)  # CLS token


x = torch.stack([get_bert_embedding(word) for word in vocab])  # [num_nodes, 768]

# 7. Constructing PyG graph data objects
graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

graph_data.vocab = vocab
graph_data.word2idx = word2idx

print("finish build graph")
graph_data.to("cuda:3")

# 8. GraphSAGE for co-occurrence embedding
model = CooccurGraphSAGE(in_channels=768, hidden_channels=256, out_channels=768)
model.to("cuda:3")
out = model(graph_data.x, graph_data.edge_index)  
print("out:", out.shape)

# Save the co-occurrence embeddings
out = out.cpu()
with open('./co_occurrence.pkl', 'wb') as f:
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
print("saved!")
