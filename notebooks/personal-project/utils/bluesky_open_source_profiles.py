"""
Helper script to discover Bluesky profiles that look interested in open
source development. The script:

1. Logs into the Bluesky API (AT Protocol) using an app password.
2. Searches for actors with open-source related keywords in their profile.
3. Scores and sorts the matches, then prints the top N profiles.

Credentials:
    - Set BLUESKY_HANDLE and BLUESKY_APP_PASSWORD env vars, or
    - Pass --handle/--app-password CLI flags (env vars take precedence).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from pydantic import BaseModel, Field

from atproto import Client, models


DEFAULT_KEYWORDS = [
    "open source",
    "oss",
    "open-source",
    "maintainer",
    "contributor",
    "opensource",
    "web3"
]


@dataclass
class ProfileHit:
    did: str
    handle: str
    display_name: str | None
    description: str | None
    followers: int | None
    score: int


def load_credentials(handle: str | None, app_password: str | None, *, verbose: bool = True) -> tuple[str, str]:
    handle = handle or os.environ.get("BLUESKY_HANDLE")
    app_password = app_password or os.environ.get("BLUESKY_APP_PASSWORD")
    if not handle or not app_password:
        msg = (
            "Missing Bluesky credentials. Provide handle/app_password arguments "
            "or set BLUESKY_HANDLE and BLUESKY_APP_PASSWORD."
        )
        raise ValueError(msg)
    if verbose:
        print(f"[setup] Using handle '{handle}' (app password from env/args).")
    return handle, app_password


def keyword_score(text: str, keywords: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for kw in keywords if kw in lowered)


def search_profiles(
    client: Client,
    keywords: Sequence[str],
    per_keyword_limit: int,
    *,
    verbose: bool = True,
) -> list[ProfileHit]:
    seen: dict[str, ProfileHit] = {}
    for keyword in keywords:
        if verbose:
            print(f"[search] Querying keyword '{keyword}' (limit={per_keyword_limit})...")
        params = models.AppBskyActorSearchActors.Params(q=keyword, limit=per_keyword_limit)
        response = client.app.bsky.actor.search_actors(params)
        actors = response.actors or []
        if verbose:
            print(f"[search] Received {len(actors)} actors for keyword '{keyword}'.")
        kept = 0
        for actor in actors:
            description = actor.description or ""
            display_name = actor.display_name or ""
            score = keyword_score(f"{display_name}\n{description}", keywords)
            if score == 0:
                continue
            hit = ProfileHit(
                did=actor.did,
                handle=actor.handle,
                display_name=actor.display_name,
                description=actor.description,
                followers=getattr(actor, "followers_count", None),
                score=score,
            )
            existing = seen.get(actor.did)
            if existing is None or (score, hit.followers or 0) > (existing.score, existing.followers or 0):
                seen[actor.did] = hit
                kept += 1
        if verbose:
            print(f"[search] Added/updated {kept} actors for keyword '{keyword}'.")
    if verbose:
        print(f"[search] Total unique profiles collected: {len(seen)}")
    return list(seen.values())


def format_profile(hit: ProfileHit) -> str:
    name = hit.display_name or ""
    followers = hit.followers if hit.followers is not None else "?"
    description = (hit.description or "").replace("\n", " ").strip()
    if len(description) > 140:
        description = description[:137] + "..."
    return (
        f"{hit.handle:<25} score={hit.score:<2} followers={followers:<6} "
        f"name='{name}' desc='{description}'"
    )


def fetch_open_source_profiles(
    *,
    keywords: Sequence[str] | None = None,
    limit: int = 10,
    per_keyword_limit: int = 25,
    handle: str | None = None,
    app_password: str | None = None,
    verbose: bool = True,
) -> list[dict[str, object]]:
    """
    Return a ranked list of Bluesky profiles that look interested in open source.

    The returned list is ready to display inside notebooks, e.g.:
    [
        {
            "rank": 1,
            "handle": "example.bsky.social",
            "display_name": "Example Dev",
            "description": "...",
            "followers": 1234,
            "score": 4,
            "did": "did:plc:123",
        },
        ...
    ]
    """
    if keywords is None:
        keywords = DEFAULT_KEYWORDS

    if verbose:
        print("[main] Starting Bluesky open-source profile discovery...")
        print(f"[main] Keywords: {list(keywords)}")
        print(f"[main] Output limit: {limit}, per-keyword limit: {per_keyword_limit}")

    handle, app_password = load_credentials(handle, app_password, verbose=verbose)
    client = Client()
    if verbose:
        print("[auth] Logging into Bluesky...")
    client.login(handle, app_password)
    if verbose:
        print("[auth] Login successful.")

    hits = search_profiles(client, keywords, per_keyword_limit, verbose=verbose)
    hits.sort(key=lambda h: (h.score, h.followers or 0), reverse=True)
    if verbose:
        print(f"[results] Sorted {len(hits)} profiles by score/followers.")

    top_hits = hits[:limit]
    if not top_hits and verbose:
        print("No profiles found. Try different keywords or raise per-keyword limit.")

    result = []
    for idx, hit in enumerate(top_hits, start=1):
        if verbose:
            print(f"{idx:02d}. {format_profile(hit)}")
        result.append(
            {
                "rank": idx,
                "did": hit.did,
                "handle": hit.handle,
                "display_name": hit.display_name,
                "description": hit.description,
                "followers": hit.followers,
                "score": hit.score,
            }
        )

    return result


class GuildCandidate(BaseModel):
    handle: str
    display_name: str | None
    description: str | None
    followers: int | None
    score: float = Field(..., ge=0.0, le=1.0)
    web3_bonus: bool
    reasoning: str
    suggested_intro: str | None = None


def analyze_profiles_for_the_guild(
    profiles: Sequence[dict[str, object]],
    openai_client,
    *,
    top_n: int = 20,
    project_description: str | None = None,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> list[dict[str, object]]:
    """
    Use ChatGPT to score which profiles are most likely to contribute to The Guild.

    Args:
        profiles: Output from fetch_open_source_profiles (dicts with handle, etc.)
        openai_client: Instructor-wrapped OpenAI client (chat.completions interface)
        top_n: Number of recommendations to return
        project_description: Optional custom project blurb
        model: OpenAI model name
        verbose: Whether to print progress information
    """
    if not profiles:
        if verbose:
            print("[guild] No profiles supplied for analysis.")
        return []

    if openai_client is None:
        raise ValueError("openai_client is required for guild analysis.")

    project_description = project_description or (
        "The Guild builds high-quality OSS tooling for GraphQL, API "
        "developer experience, and the broader Web3 + infra ecosystem. "
        "They value engineers who are active in open source, enjoy API "
        "tooling, and experiment with decentralized or Web3 technologies."
    )

    trimmed_profiles = profiles[:100]
    if verbose:
        print(f"[guild] Preparing {len(trimmed_profiles)} profiles for AI analysis (top_n={top_n}).")

    profile_lines = []
    for prof in trimmed_profiles:
        desc = str(prof.get("description") or "").replace("\n", " ")
        if len(desc) > 240:
            desc = desc[:237] + "..."
        line = (
            f"- @{prof.get('handle')} ({prof.get('display_name') or prof.get('handle')}) | "
            f"followers={prof.get('followers', '?')} | desc={desc}"
        )
        profile_lines.append(line)

    prompt = f"""You are evaluating Bluesky profiles to recruit contributors for The Guild,
an open-source organization known for GraphQL, API tooling, and developer productivity projects.

Project context:
{project_description}

Profiles to consider:
{chr(10).join(profile_lines)}

Please return the top {top_n} candidates who are most likely to collaborate on The Guild's OSS initiatives.
Prioritize people who explicitly mention open source, developer tooling, GraphQL, APIs, or Web3/crypto.
Set web3_bonus=True if their bio clearly references Web3, crypto, blockchain, or decentralized tech.
Respond with JSON objects including handle, display_name, description, followers, score (0-1),
web3_bonus, reasoning, and a brief suggested_intro referencing their interests."""

    try:
        if verbose:
            print("[guild] Sending profiles to OpenAI for ranking...")
        response: List[GuildCandidate] = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert open-source community scout focused on recruiting for "
                        "The Guild's developer tooling projects."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_model=List[GuildCandidate],
            temperature=0.25,
        )
    except Exception as exc:
        print(f"[guild] ❌ Error while calling OpenAI: {exc}")
        import traceback

        traceback.print_exc()
        return []

    ranked = []
    for idx, candidate in enumerate(response[:top_n], start=1):
        entry = {
            "rank": idx,
            "handle": candidate.handle,
            "display_name": candidate.display_name,
            "description": candidate.description,
            "followers": candidate.followers,
            "score": candidate.score,
            "web3_bonus": candidate.web3_bonus,
            "reasoning": candidate.reasoning,
            "suggested_intro": candidate.suggested_intro,
        }
        ranked.append(entry)

    if verbose:
        print(f"[guild] Received {len(ranked)} ranked candidates from OpenAI.")

    return ranked

