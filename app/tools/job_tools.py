import os
import logging
import asyncpg
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


_db_pool = None


async def get_db_pool():
    """
    Initializes and returns the asyncpg connection pool.
    """
    global _db_pool
    if _db_pool is None:
        try:
            _db_pool = await asyncpg.create_pool(
                os.getenv("DATABASE_URL"),
                min_size=1,
                max_size=10,  # Pool keeps up to 10 connections alive
            )
            logger.info("PostgreSQL connection pool created successfully.")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    return _db_pool


async def search_jobs(location: str, limit: int = 5) -> list[dict]:
    """
    Searches the Nigerian jobs vault by city or region.
    Case-insensitive LIKE search against the location column.

    Parameters:
        location (str): City or region e.g. Lagos, Abuja, Remote
        limit (int): Maximum results to return. Default 5.

    Returns:
        list[dict]: Job listings with title, company, location, salary, job_url
    """
    pool = await get_db_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                    SELECT title, company, location, salary, job_url
                    FROM jobs
                    WHERE LOWER(location) LIKE LOWER($1)
                    ORDER BY scraped_at DESC
                    LIMIT $2
                """,
                f"%{location}%",
                limit,
            )
        results = [dict(row) for row in rows]
        logger.info(f"search_jobs('{location}') → {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"search_jobs error: {e}")
        return []


async def get_latest_jobs(limit: int = 5) -> list[dict]:
    """
    Returns the most recently scraped jobs from the vault.
    Used when user asks for latest or recent jobs without specifying a location.

    Parameters:
        limit (int): Number of jobs to return. Default 5.

    Returns:
        list[dict]: Most recent job listings ordered by scraped_at DESC
    """
    pool = await get_db_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT title, company, location, salary, job_url
                FROM jobs
                ORDER BY scraped_at DESC
                LIMIT $1
            """,
                limit,
            )

            results = [dict(row) for row in rows]
            logger.info(f"get_latest_jobs() → {len(results)} results")
            return results

    except Exception as e:
        logger.error(f"get_latest_jobs error: {e}")
        return []


async def get_jobs_by_keyword(keyword: str, limit: int = 10) -> list[dict]:
    """
    Searches jobs by keyword in the job title field.
    Used when user asks for a specific role type e.g. 'engineer jobs'.

    Parameters:
        keyword (str): Title keyword e.g. engineer, accountant, manager
        limit (int): Maximum results. Default 10.

    Returns:
        list[dict]: Jobs whose titles contain the keyword
    """
    pool = await get_db_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT title, company, location, salary, job_url
                FROM jobs
                WHERE LOWER(title) LIKE LOWER($1)
                ORDER BY scraped_at DESC
                LIMIT $2
            """,
                f"%{keyword}%",
                limit,
            )

            results = [dict(row) for row in rows]
            logger.info(f"get_jobs_by_keyword('{keyword}') → {len(results)} results")
            return results

    except Exception as e:
        logger.error(f"get_jobs_by_keyword error: {e}")
        return []


async def get_vault_stats() -> dict:
    """
    Returns summary statistics about the jobs database.
    Used when user asks how many jobs are available or when data was last updated.

    Returns:
        dict with keys:
            total_jobs (int): Total rows in the jobs table
            last_scrape (str): ISO timestamp of the most recent scraped_at value
    """
    pool = await get_db_pool()
    try:
        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM jobs")
            last_scrape = await conn.fetchval("SELECT MAX(scraped_at) FROM jobs")

            return {
                "total_jobs": total,
                "last_scrape": str(last_scrape) if last_scrape else "Never",
            }

    except Exception as e:
        logger.error(f"get_vault_stats error: {e}")
        return {"total_jobs": 0, "last_scrape": "Unknown"}
