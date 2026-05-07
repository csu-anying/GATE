
import json
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load Model
model_id = "your_path/OpenBioLLM-Llama3-8B/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# 2. Loading external knowledge bases
docs_path = "docs.json"  # your path
try:
    with open(docs_path, "r", encoding="utf-8") as f:
        docs_data = json.load(f)
        docs = [v for v in docs_data.values()]  # format： {disease: "description"}
except:
    with open(docs_path, "r", encoding="utf-8") as f:
        docs = f.readlines()

# 3. Constructing vector indexes (FAISS)
emb_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = emb_model.encode(docs, normalize_embeddings=True)
index = faiss.IndexFlatIP(doc_embs.shape[1])
index.add(doc_embs)

def retrieve_knowledge(query, top_k=3):
    """Searching for relevant knowledge based on disease names"""
    q_emb = emb_model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    return [docs[i] for i in I[0]]

# 4. Constructing prompt
def build_prompt(disease):
    system_prompt = (
        "You are a clinical assistant generating professional-style ECG interpretation text for a given heart disease label. "
        "Your description should resemble a brief but accurate report entry, like those used in real ECG summaries."
    )

    retrieved = retrieve_knowledge(disease, top_k=3)
    knowledge = "\n".join(retrieved)

    example_description = (
        "'NDT': 'Non-diagnostic T-wave abnormalities, including mild T-wave flattening and inversions observed in leads I, II, III, aVL, and aVF. "
        "These changes are often nonspecific and may not indicate acute pathology.'"
    )

    user_prompt = (
        "Given a heart disease label, write a brief report-style ECG description (within 50 words), "
        "combining the typical clinical expression, possible variations or synonyms (without listing them separately), and characteristic waveform findings. "
        "Write it in natural report style, as found in professional ECG readings. Avoid using labels like 'synonyms' or 'ECG:'.\n\n"
        f"Example:\n{example_description}\n\n"
        f"Disease: {disease}\n\n"
        f"Relevant knowledge from expert resources:\n{knowledge}\n\n"
    )

    return f"[SYSTEM]: {system_prompt}\n[USER]: {user_prompt}\n[ASSISTANT]:"

# 5. Inference and Output
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

input_path = "input.json"
output_path = "output_descriptions.json"

with open(input_path, "r", encoding="utf-8") as f:
    disease_dict = json.load(f)

output_dict = {}

for disease in tqdm(disease_dict.keys()):
    prompt = build_prompt(disease)
    output = pipeline(
        prompt,
        max_new_tokens=128,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    result = output[0]["generated_text"][len(prompt):].strip()

    output_dict[disease] = result

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_dict, f, indent=2, ensure_ascii=False)

print(f"saved {output_path}")

