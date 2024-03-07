import os

os.chdir("/home/tommaso/Repositories/teleRAG/")


def load_api_token(path='bot/API_token'):
    """Load API token string from file"""
    if os.path.isfile(path):
        with open(path) as file:
            return file.read()
    else:
        return None


def check_length(text: str, max_length: int = 4096):
    """Check length of text, return True if text is longer than max length"""
    return len(text) >= max_length


def split_text(text: str, max_length: int = 4000):
    """Split text into chunks if too long"""
    chunks = []
    for x in range(0, len(text), max_length):
        chunks.append(text[x:x + max_length] + " [" + str(int(1 + x / max_length)) + "/" + str(int(1 + len(text) / max_length)) + "]")
    return chunks
