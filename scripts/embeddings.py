from transformers import AutoTokenizer, AutoModel
import torch
import os
from sentence_transformers import SentenceTransformer

os.chdir("/home/tommaso/Repositories/spawn-nlp/")


class SentenceEmbedder:
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path, cache_folder="./models/cache")

    def encode(self, sentences):
        return self.model.encode(sentences)

#### TEST ####
# Sentence Embedder without the sentence_transformers dependecy
# DOES NOT WORK, still trying to figure out why
class SentenceEmbedderHF:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="./models/cache")
        self.model = AutoModel.from_pretrained(model_path, cache_dir="./models/cache")

    # Mean Pooling - Take attention mask into account for correct averaging
    def __mean_pooling__(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, max_length=128, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Perform pooling. In this case, mean pooling.
        sentence_embeddings = self.__mean_pooling__(model_output, encoded_input['attention_mask'])
        return sentence_embeddings
