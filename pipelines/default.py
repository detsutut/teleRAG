import os

from scripts.embedder.embeddings import SentenceEmbedder

os.chdir("/home/tommaso/Repositories/teleRAG/")

def main():
    sentences = ["Questo è un esempio di frase", "Questo è un ulteriore esempio"]
    embedder = SentenceEmbedder('efederici/sentence-IT5-base')
    embeddings = embedder.encode(sentences)
    print(embeddings)

if __name__ == "__main__":
    main()