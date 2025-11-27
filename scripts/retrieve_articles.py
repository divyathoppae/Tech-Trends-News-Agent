import os
import requests
import logging
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    filename="retrieve.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------
# Load API key
# ----------------------------
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

# Use 'everything' endpoint to allow date filtering
BASE_URL = "https://newsapi.org/v2/everything"


# ----------------------------
# Fetch articles function
# ----------------------------
def fetch_articles(query="technology", language="en"):
    # Calculate date range: past 30 days
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": language,
        "apiKey": API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok" or not data.get("articles"):
            logging.warning(f"No articles found for query: {query}")
            return None

        logging.info(f"Retrieved {len(data['articles'])} articles for query: {query} (from {from_date} to {to_date})")
        return data

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return None


# ----------------------------
# Save articles function
# ----------------------------
def save_articles(data, query):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw JSON
    raw_path = f"../data/raw/{query}_{timestamp}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Save cleaned text (source, title, description)
    clean_path = f"../data/clean/{query}_{timestamp}.txt"
    with open(clean_path, "w", encoding="utf-8") as f:
        for article in data["articles"]:
            source = article.get("source", {}).get("name", "")
            title = article.get("title", "")
            description = article.get("description", "")
            f.write(f"Source: {source}\nTitle: {title}\nDescription: {description}\n\n")

    logging.info(f"Saved raw to {raw_path} and cleaned to {clean_path}")


# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    query = "technology"
    articles = fetch_articles(query)
    if articles:
        save_articles(articles, query)
