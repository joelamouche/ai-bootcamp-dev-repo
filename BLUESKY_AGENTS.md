## The Guild Bluesky Agent System Plan

This document captures the scope for the Bluesky-focused final project, building on earlier bootcamp notebooks:

- `notebooks/personal-project/03-bluesky-api-research.ipynb` – API exploration, authentication flow, follower discovery, and early AI-assisted ranking experiments.
- `notebooks/personal-project/04-bluesky-lib-functions.ipynb` – Reusable helpers (`fetch_followers_from_user`, `analyze_users_with_ai`, `follow_top_users`, `search_profiles_by_keyword`, etc.) plus LangGraph-ready components.

---

## Objectives

- **Grow Guild presence** – add ~5–10 high-relevance accounts per day, ramping up during onboarding.
- **Understand the network** – analyze followers/following, posts, and trending topics to refine targeting.
- **Curate knowledge** – collect and vectorize noteworthy accounts, posts, and long-form content (Medium/Substack) for downstream retrieval.
- **Publish with HITL** – surface daily post suggestions tied to Guild activity (GitHub + Discord feed) and allow humans to schedule/approve.

Existing tooling from the notebooks already covers:
- Profile/follower fetches, activity filtering, and AI-based ranking (03 + 04).
- Keyword search across Bluesky posts.
- Structured AI reasoning pipelines using Instructor + OpenAI.

**Note**: The batch analysis function `analyze_users_with_ai` will be replaced with a per-profile analysis approach (see Core Profile Analysis below).

---

## Core Profile Analysis Function

The foundation of the system is a **single-profile analysis function** that processes one profile at a time:

**`analyze_profile(profile_handle)`**:
- Fetches profile data (description, recent posting activity, follower/following counts)
- Uses AI (Instructor + OpenAI) to generate:
  - **Summary**: Concise profile summary including description and recent activity patterns
  - **Keywords**: Extracted relevant keywords/topics
  - **Relevance Score**: Likelihood to follow (0.0-1.0) based on alignment with Guild interests
- Embeds the summary using OpenAI text-embedding-3-large
- Upserts into Qdrant `profiles` collection with fields:
  - `handle`, `display_name`, `description`, `summary`, `keywords`, `relevance_score`
  - `already_followed` (boolean), `last_scanned` (timestamp)
  - `embedding` (vector)
- **If relevance_score > threshold**: 
  - Fetches recent posts from the profile
  - Analyzes and embeds relevant posts into Qdrant `content` collection
  - Posts include metadata: `poster_handle`, `date`, `media_urls`, `text`, `embedding`

This per-profile approach replaces batch analysis (`analyze_users_with_ai`) and enables:
- Incremental processing (process profiles one at a time, respect rate limits)
- Simple filtering for Growth Agent: query Qdrant for `already_followed=False AND relevance_score > threshold`
- RAG-based content generation: Post Agent queries `content` collection for relevant posts/articles

---

## High-Level Agent Topology

Guidance from the bootcamp mentor Q&A: keep agents single-purpose until tool count > 5, then split along clear domain boundaries. Applying that advice yields three collaborating LangGraph (orchestrated) agents, each with ≤4 tools.

| Agent | Responsibilities | Primary Tools / Functions | Notes |
| --- | --- | --- | --- |
| **Analyzer Agent** | 1) Inspect user profiles/posts/followers 2) Analyze and persist them | `analyze_profile`, `fetch_followers_from_user`, `search_profiles_by_keyword`, Qdrant profile/content DB upserts | Runs as a scheduled job (e.g. Airflow / Cloud Composer DAG) every few hours to stay within rate limits. Processes profiles one at a time. |
| **Growth Agent** | 2) Follow relevant users | Qdrant profile queries (filter: `already_followed=False AND relevance_score > threshold`), `follow_top_users` | Can be triggered after Analyzer completes, consuming the fresh embeddings to choose follow targets (HITL optional). Unfollow functionality postponed to later. |
| **Post Agent** | 3) Draft posts using Guild activity feed + vectorized content, coordinate HITL approvals | Guild activity feed fetcher, Tavily web search wrapper, content RAG queries, scheduler/defer queue | Outputs multiple draft posts per slot, stores them in a review queue (Google Doc or lightweight dashboard) before publishing. |

> If the entire workflow ever exceeds five tools again, split individual agents further (e.g., dedicated “Research” vs “Posting” nodes inside LangGraph).

---

## Data & Infrastructure Plan

| Need | Approach |
| --- | --- |
| **Vector DB** | Qdrant collections: `profiles` (profile summaries + embeddings, with `already_followed`, `last_scanned` fields) and `content` (chunks from Bluesky posts, tweets, and articles with metadata: `poster_handle`, `date`, `media_urls`, `text`, `embedding`). Analyzer Agent owns writes; Growth/Post agents read via RAG. |
| **Scheduler** | Use Airflow/Cloud Composer for recurring Analyzer runs, downstream Growth tasks, and eventual Post Agent triggers. Local cron acceptable for prototyping. |
| **LangGraph Runtime** | Python-first orchestration tying existing notebook functions into graph nodes; each agent is its own LangGraph subgraph. |
| **Human-in-the-loop UI** | Minimal viable option: Google Doc or Notion table where Post Agent drops suggested posts + metadata. Later upgrade to a small FastAPI dashboard storing drafts + approval status. |

---

## Roadmap

1. **Experiment with Single Profile Analysis (Jupyter Notebooks)**  
   - Build and test `analyze_profile(profile_handle)` function in notebooks.  
   - Experiment with AI prompts to generate summaries, extract keywords, and compute relevance scores.  
   - Test embedding generation and Qdrant upserts for both profiles and their posts.  
   - Validate filtering logic: query Qdrant for `already_followed=False AND relevance_score > threshold`.  
   - Compare results with existing `analyze_users_with_ai` approach to validate the per-profile strategy.

2. **Harden Notebook Utilities**  
   - Move the reusable pieces from `03`/`04` notebooks into `notebooks/personal-project/utils/` (already partially done).  
   - Add docstrings/tests where missing (e.g., `search_profiles_by_keyword` still needs a `UserProfile` model).  
   - Refactor to use `analyze_profile` instead of `analyze_users_with_ai` where appropriate.

3. **Stand Up Qdrant + Data Contracts**  
   - Define `ProfileDoc` schema: `id`, `handle`, `display_name`, `description`, `summary`, `keywords`, `relevance_score`, `already_followed`, `last_scanned`, `embedding`.  
   - Define `ContentDoc` schema: `id`, `poster_handle`, `date`, `media_urls`, `text`, `source_type` (post/tweet/article), `embedding`.  
   - Implement embedding generation (OpenAI text-embedding-3-large or similar) inside Analyzer Agent.

4. **Implement Analyzer Agent (LangGraph node)**  
   - Inputs: target handles/keywords, freshness thresholds.  
   - Process profiles one at a time using `analyze_profile`.  
   - For relevant profiles, fetch and embed posts into `content` collection.  
   - Outputs: upserted Qdrant vectors + activity logs.  
   - Schedule via Airflow; include Tavily scraping for article summaries when necessary.

5. **Implement Growth Agent**  
   - Query Qdrant profiles collection: `already_followed=False AND relevance_score > threshold`.  
   - Sort by relevance_score, respect follow limits (e.g., max 10/day).  
   - Use `follow_top_users` helper (updated to work with Qdrant query results).  
   - Update `already_followed=True` in Qdrant after successful follows.  
   - Provide manual review option (list with relevance scores).  
   - **Note**: Unfollow functionality postponed to later.

6. **Implement Post Agent**  
   - Gather Guild activity feed snapshots, related content (via Qdrant `content` collection RAG), and Tavily articles.  
   - Generate 2–3 candidate posts per slot; store in HITL queue.  
   - Once approved, call Bluesky posting endpoint (future extension).

7. **End-to-End DAG**  
   - Airflow DAG: `analyzer` → `growth` → `post-drafts`.  
   - Add observability (logging, OpenTelemetry).  
   - Document operational runbooks (rate limits, retries).

8. **Stretch Goals**  
   - Integrate Discord/GitHub activity feed refresh tool so Post Agent can pull latest context automatically.  
   - Add Tavily-powered "topic radar" to Analyzer to enrich the `content` collection with off-platform articles.  
   - Experiment with LangGraph checkpoints to resume partially completed pipelines.  
   - Implement unfollow heuristics for stale or low-engagement accounts.

---

## Tech Stack Summary

- **Language**: Python (per bootcamp tooling).  
- **Agents/Workflow**: LangGraph for orchestration, Instructor + OpenAI for structured outputs.  
- **Data Layer**: Qdrant for embeddings (`profiles` and `content` collections), existing `utils` package for API wrappers.  
- **Scheduling**: Airflow/Cloud Composer (Analyzer & Growth) + manual cron fallback.  
- **External APIs**: Bluesky (ATProto client), Tavily search, Guild activity feed service.  
- **HITL Surface**: Google Doc / Notion initially; optional FastAPI dashboard later.

This plan keeps each agent focused (<5 tools), uses per-profile analysis for better incremental processing and filtering, reuses the notebook groundwork, and positions the project for incremental delivery while staying within Bluesky's cost-friendly environment.


