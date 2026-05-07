import numpy as np
import pandas as pd
from transformers import pipeline
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from wordcloud import WordCloud

def extract_clean_entities(text, ner_pipeline):
    try:
        results = ner_pipeline(text)
        entities = set()
        for ent in results:
            entity = text[ent['start']:ent['end']].strip().lower()
            entities.add(entity)
        return list(entities)
    except Exception as e:
        print("Entity extraction failed:", e)
        return []


# Step 1: Load reports
df = pd.read_excel("report_trans.xlsx")
reports = df["report"].dropna().tolist()
reports = reports[0:50]

# Step 2: Initialize NER pipeline
ner_pipeline = pipeline(
    "ner",
    model="/home/tangsy/code/biomedical-ner-all/",
    aggregation_strategy="simple"
)

# Step 3: Extract entities from each report
all_entities_list = []  

print("Extracting entities from reports...")
for text in tqdm(reports):
    try:
        entity_words = extract_clean_entities(text, ner_pipeline)
        print(entity_words)
        all_entities_list.append(entity_words)
    except Exception as e:
        print("NER failed:", e)
        all_entities_list.append([])


# Step 4: Count entity frequencies for word cloud
entity_freq = defaultdict(int)
for entities in all_entities_list:
    for entity in entities:
        entity_freq[entity] += 1

# Step 5: Generate and save word cloud
fig, ax = plt.subplots(1, 1, figsize=(12, 10))  

wordcloud = WordCloud(
    width=1200,           
    height=1000,          
    background_color='white',
    colormap='viridis',
    max_words=200,
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
).generate_from_frequencies(entity_freq)

ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.set_title("Entity Frequency Word Cloud", fontsize=16, pad=20)

plt.tight_layout()

plt.savefig("wordcloud_output.pdf", format='pdf', dpi=300, bbox_inches='tight')
print("图片已保存为 wordcloud_output.pdf")

plt.show()