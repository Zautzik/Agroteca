"""Quota guards for the public demo (profile=hosted; see migrations/003_public_demo.sql).

Each question is counted BEFORE it is answered -- once against a global per-UTC-day cap and once
against a per-IP per-hour cap -- with an atomic upsert, so concurrent requests can't race past a
limit. A cap of 0 disables that guard (the local default). IPs are stored only as a salted
SHA-256, never raw.
"""
import hashlib

from agroteca.config import settings

_DAILY_SQL = """
    INSERT INTO usage_daily (day, requests) VALUES ((now() AT TIME ZONE 'utc')::date, 1)
    ON CONFLICT (day) DO UPDATE SET requests = usage_daily.requests + 1
    RETURNING requests
"""

_IP_SQL = """
    INSERT INTO usage_ip_hourly (ip_hash, hour, requests)
    VALUES (%s, date_trunc('hour', now()), 1)
    ON CONFLICT (ip_hash, hour) DO UPDATE SET requests = usage_ip_hourly.requests + 1
    RETURNING requests
"""


def hash_ip(ip: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()


def client_ip(headers, fallback: str | None) -> str:
    """The visitor's address: the first X-Forwarded-For hop (set by the hosting edge), else the
    socket peer. Only ever used hashed."""
    forwarded = headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    return first or headers.get("x-real-ip", "") or fallback or "unknown"


async def admit(conn, ip: str) -> str | None:
    """Count this question; return None if it may proceed, else a short reason it may not."""
    if settings.daily_request_cap > 0:
        cur = await conn.execute(_DAILY_SQL)
        if (await cur.fetchone())[0] > settings.daily_request_cap:
            return "daily_cap"
    if settings.per_ip_hourly_cap > 0:
        cur = await conn.execute(_IP_SQL, (hash_ip(ip, settings.ip_hash_salt),))
        if (await cur.fetchone())[0] > settings.per_ip_hourly_cap:
            return "ip_cap"
    return None
