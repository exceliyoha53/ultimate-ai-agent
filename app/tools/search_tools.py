import logging
import os
import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


http_client = httpx.AsyncClient(timeout=10.0)


async def search_web(query: str, num_results: int = 5) -> list[dict]:
    """
    Searches the web using Serper API (Google Search).
    Use for current events, news, general knowledge questions,
    anything not in the local database.

    Parameters:
        query (str): Search query
        num_results (int): Number of results to return

    Returns:
        list[dict]: Search results with title, link, snippet
    """
    try:
        response = await http_client.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": os.getenv("SERPER_API_KEY"),
                "Content-Type": "application/json",
            },
            json={"q": query, "num": num_results},
        )
        response.raise_for_status()

        data = response.json()
        results = []
        for item in data.get("organic", []):
            results.append(
                {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                }
            )

        logger.info(f"search_web({query}) returned {len(results)} results")
        return results
    
    except Exception as e:
        logger.error(f"search_web error: {e}")
        return []


async def get_weather(city: str) -> dict:
    """
    Gets current weather for a city.
    Use when user asks about weather.

    Parameters:
        city (str): City name e.g. Kaduna, Lagos, Abuja

    Returns:
        dict: Temperature, condition, humidity
    """
    try:
        response = await http_client.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": os.getenv("OPENWEATHER_API_KEY"),
                "units": "metric",
            },
        )
        response.raise_for_status()

        data = response.json()
        if data.get("cod") != 200:
            return {"error": f"City not found: {city}"}
        
        return {
            "city": city,
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "condition": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
        }
    
    except Exception as e:
        logger.error(f"get_weather error: {e}")
        return {"error": str(e)}


async def get_news(topic: str, num_articles: int = 5) -> list[dict]:
    """
    Gets latest news articles on a topic.
    Use when user asks about current events or news.

    Parameters:
        topic (str): News topic e.g. Nigeria economy, tech jobs
        num_articles (int): Number of articles to return

    Returns:
        list[dict]: Articles with title, description, url, publishedAt
    """
    try:
        response = await http_client.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": topic,
                "apiKey": os.getenv("NEWS_API_KEY"),
                "pageSize": num_articles,
                "sortBy": "publishedAt",
                "language": "en",
            },
        )
        response.raise_for_status()

        data = response.json()
        articles = []
        for article in data.get("articles", []):
            articles.append(
                {
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "published": article.get("publishedAt"),
                }
            )

        logger.info(f"get_news({topic}) returned {len(articles)} articles")
        return articles
    
    except Exception as e:
        logger.error(f"get_news error: {e}")
        return []
