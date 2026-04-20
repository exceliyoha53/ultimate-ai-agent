import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from app.api.routes import router
from app.memory.redis_memory import redis_client
from app.tools.search_tools import http_client as search_client

load_dotenv()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await redis_client.ping()
        logger.info("Connected to Redis successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")

    yield

    await redis_client.aclose()
    await search_client.aclose()
    logger.info("Shutdown complete: Connections closed.")


app = FastAPI(
    title="Ultimate AI Agent",
    description=(
        "Production-grade AI agent with web search, Nigerian job intelligence, "
        "email sending, weather, news, voice responses, and persistent memory. "
        "Powered by Gemini 2.5 Flash + LangGraph + Redis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(router)



@app.get("/health")
async def health():
    return {
        "status": "online",
        "agent": "Ultimate AI Agent",
        "tools": [
            "web_search",
            "weather",
            "news",
            "email",
            "memory",
            "nigerian_jobs",
            "voice",
        ],
    }
