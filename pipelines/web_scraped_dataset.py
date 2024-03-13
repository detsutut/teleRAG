import time
import os
import pandas as pd
from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import fetch_url, extract, extract_metadata

os.chdir("/home/tommaso/Repositories/teleRAG/")

def get_urls_from_sitemap(resource_url: str) -> list:
    """
    Funzione che recupera la sitemap attraverso Trafilatura
    """
    urls = sitemap_search(resource_url)
    return urls


def create_dataset(list_of_websites: list) -> pd.DataFrame:
    """
    Funzione che crea un DataFrame Pandas di URL e articoli.
    """
    data = []
    for website in tqdm(list_of_websites, desc="Websites"):
        urls = get_urls_from_sitemap(website)
        for url in tqdm(urls, desc="URLs"):
            html = fetch_url(url)
            body = extract(html)
            try:
                metadata = extract_metadata(html)
                title = metadata.title
                description = metadata.description
            except:
                title = ""
                description = ""
            d = {
                'url': url,
                "body": body,
                "title": title,
                "description": description
            }
            data.append(d)
            time.sleep(0.5)
    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    df = df.dropna()

    return df


if __name__ == "__main__":
    list_of_websites = [
        "https://cats.com/",
        "https://en.wikipedia.org/wiki/Cat"
    ]
    df = create_dataset(list_of_websites)
    df.to_csv("./data/ws_dataset.csv", index=False)
