import os
import re
import json
import logging
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    filename="retrieve.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------
# Ensure NLTK resources
# ----------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ----------------------------
# Load API key
# ----------------------------
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

BASE_URL = "https://newsapi.org/v2/everything"

# ----------------------------
# Fetch one page of articles
# ----------------------------
def fetch_articles(query="technology", language="en", page_size=100, page=1):
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": language,
        "pageSize": page_size,
        "page": page,
        "apiKey": API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "ok" or not data.get("articles"):
            logging.warning(f"No articles found for query: {query}, page {page}")
            return None
        logging.info(f"Retrieved {len(data['articles'])} articles for query: {query}, page {page}")
        return data["articles"]
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return None

# ----------------------------
# Tokenize helper
# ----------------------------
def clean_text(text: str):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if re.match(r"[a-zA-Z0-9]+", t)]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# ----------------------------
# Fetch full article text
# ----------------------------
def fetch_article_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text() for p in paragraphs)
    except Exception as e:
        logging.warning(f"Failed to fetch {url}: {e}")
        return ""

# ----------------------------
# Save raw + preprocessed
# ----------------------------
def save_articles(all_articles, query):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("../data/raw", exist_ok=True)
    os.makedirs("../data/processed", exist_ok=True)

    # Save raw JSON
    raw_path = os.path.join("..","data", "raw", f"{query}_{timestamp}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2)

    # Preprocess each article
    processed = []
    for idx, art in enumerate(all_articles, start=1):
        url = art.get("url")
        title = art.get("title", f"Article {idx}")
        text = fetch_article_text(url)
        tokens = clean_text(text)
        processed.append({
            "id": f"article_{idx}",
            "url": url,
            "title": title,
            "tokens": tokens
        })

    # Save processed JSON
    processed_path = os.path.join("..", "data", "processed", f"{query}_{timestamp}.json")
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)

    logging.info(f"Saved raw to {raw_path} and processed to {processed_path}")
    print(f"✅ Saved {len(processed)} processed articles to {processed_path}")

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    query = "technology"
    all_articles = []
    for page in range(1, 4):  # up to 3 pages (~300 articles)
        articles = fetch_articles(query, page=page)
        if articles:
            all_articles.extend(articles)

    if all_articles:
        save_articles(all_articles, query)
    else:
        print("⚠️ No articles retrieved.")
