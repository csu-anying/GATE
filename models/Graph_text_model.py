import torch
import torch.nn as nn
import yaml
from einops import rearrange
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

from models.encoder.graph_encoder import Graph_Encoder, Graph_Encoder_time
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CooccurrenceAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, text_emb, graph_emb):
        # text_emb: [batch, seq_len, dim]
        # graph_emb: [vocab_size, dim]

        Q = self.query_proj(text_emb)  # [B, L, D]
        K = self.key_proj(graph_emb)  # [V, D]
        V = self.value_proj(graph_emb)  # [V, D]

        # attention scores: [B, L, V]
        attn_scores = torch.matmul(Q, K.transpose(0, 1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # attention output: [B, L, D]
        attn_output = torch.matmul(attn_weights, V)

        return self.ln(text_emb + attn_output)  # Residual + LayerNorm



class ECG_Graph_Text_with_co(torch.nn.Module):
    def __init__(self, network_config):
        super(ECG_Graph_Text_with_co, self).__init__()

        self.aug_with_cooccurrence = True
        self.c_emb = None
        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']
        # ecg_graph signal encoder
        self.graph_encoder = Graph_Encoder(network_config)
        # load text: use config value to support online model loading from Hugging Face Hub
        model_name_or_path = network_config.get('text_model', "../Clinical_ModernBERT/")
        self.lm_model = AutoModel.from_pretrained(model_name_or_path, local_files_only=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=False)

        # text projector 
        self.proj_t = nn.Sequential(
            nn.Linear(768, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )
        self.fusion = CooccurrenceAttention(hidden_dim=768)

    @torch.no_grad()
    def co_occurrence_emb(self, c_emb):
        # [1038, 768] 32,768
        self.c_emb = c_emb
        print(self.c_emb.shape)

    # The input text data is segmented into words, and then padded or truncated
    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=256,
                                                            padding='max_length',
                                                            return_tensors='pt')

        return tokenizer_output

    # Extract the embedding features of the text, process them using the BERT model, and obtain a pooled output
    @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        text_emb = self.lm_model(input_ids=input_ids,
                                 attention_mask=attention_mask).pooler_output
        return text_emb

    def forward(self, ecg, input_ids, attention_mask, mode='train'):
        # graph encoder
        if mode == 'train':
            proj_ecg_emb, ecg_graph_emb1, ecg_graph_emb2 = self.graph_encoder(ecg, mode)
        else:
            proj_ecg_emb = self.graph_encoder(ecg, mode)
        # Normalize the projection head to achieve intermodal alignment
        proj_ecg_emb = normalize(proj_ecg_emb, dim=-1)

        # get text feature
        text_emb = self.get_text_emb(input_ids, attention_mask)
        if self.aug_with_cooccurrence:
            self.c_emb = self.c_emb.to(text_emb.device)
            fuse_emb = self.fusion(text_emb, self.c_emb)
            proj_text_emb = self.proj_t(fuse_emb.contiguous())

        else:
            proj_text_emb = self.proj_t(text_emb.contiguous())
        proj_text_emb = normalize(proj_text_emb, dim=-1)

        # Returns the ECG embedding and the modally aligned embedding.
        if mode == 'train':
            return {'ecg_emb': [ecg_graph_emb1, ecg_graph_emb2],
                    'proj_ecg_emb': [proj_ecg_emb],
                    'proj_text_emb': [proj_text_emb]}
        else:
            return {'proj_ecg_emb': [proj_ecg_emb],
                    'proj_text_emb': [proj_text_emb]}


# ============================ only time=======================
class ECG_Graph_Text_only_time(torch.nn.Module):
    def __init__(self, network_config):
        super(ECG_Graph_Text_only_time, self).__init__()

        self.aug_with_cooccurrence = True
        self.c_emb = None
        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']

        # ecg_graph signal encoder
        self.time_encoder = Graph_Encoder_time(network_config)

        # load text: use config value to support online model loading from Hugging Face Hub
        model_name_or_path = network_config.get('text_model', "../Clinical_ModernBERT/")
        self.lm_model = AutoModel.from_pretrained(model_name_or_path, local_files_only=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=False)

        # text projector
        self.proj_t = nn.Sequential(
            nn.Linear(768, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )
        self.fusion = CooccurrenceAttention(hidden_dim=768)

    @torch.no_grad()
    def co_occurrence_emb(self, c_emb):
        # [1038, 768]
        self.c_emb = c_emb

    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=256,
                                                            padding='max_length',
                                                            return_tensors='pt')

        return tokenizer_output

    @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        text_emb = self.lm_model(input_ids=input_ids,
                                 attention_mask=attention_mask).pooler_output
        return text_emb

    def forward(self, ecg, input_ids, attention_mask, mode='train'):
        # graph encoder
        if mode == 'train':
            proj_ecg_emb, ecg_graph_emb1, ecg_graph_emb2 = self.time_encoder(ecg, mode)
        else:
            proj_ecg_emb = self.time_encoder(ecg, mode)

        proj_ecg_emb = normalize(proj_ecg_emb, dim=-1)

        # get text feature
        # text feature extraction is independent of the type of ecg encoder
        text_emb = self.get_text_emb(input_ids, attention_mask)
        if self.aug_with_cooccurrence:
            self.c_emb = self.c_emb.to(text_emb.device)
            fuse_emb = self.fusion(text_emb, self.c_emb)
            proj_text_emb = self.proj_t(fuse_emb.contiguous())

            # print("text_emb:{}, c_emb:{}, proj_text_emb:{}".format(text_emb.shape, self.c_emb.shape, proj_text_emb.shape))
            # co_text_emb = torch.matmul(text_emb, self.c_emb.T)
            # proj_text_emb = self.proj_t_w_c(co_text_emb.contiguous())
        else:
            proj_text_emb = self.proj_t(text_emb.contiguous())
        proj_text_emb = normalize(proj_text_emb, dim=-1)

        if mode == 'train':
            return {'ecg_emb': [ecg_graph_emb1, ecg_graph_emb2],
                    'proj_ecg_emb': [proj_ecg_emb],
                    'proj_text_emb': [proj_text_emb]}
        else:
            return {'proj_ecg_emb': [proj_ecg_emb],
                    'proj_text_emb': [proj_text_emb]}


if __name__ == "__main__":
    # Assume a batch size of 16, 12 leads, and each lead signal length of 5000.
    input_data = torch.randn(16, 12, 5000)  # Random input data for testing
    config = yaml.load(open("your_path/config.yaml", "r"), Loader=yaml.FullLoader)
    model = ECG_Graph_Text_with_co(config['network'])

    result = model(input_data, [], [])
