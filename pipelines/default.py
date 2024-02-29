from scripts.embeddings import SentenceEmbedder

def main():
    sentences = ["Questo è un esempio di frase", "Questo è un ulteriore esempio"]
    embedder = SentenceEmbedder('efederici/sentence-IT5-base')
    embeddings = embedder.encode(sentences)
    print(embeddings)

if __name__ == "__main__":
    main()