# ViralLens Discovery Provider Protocol

## 1. Executive Overview
The ViralLens Discovery Provider Protocol establishes an explicit, formal boundary between external platform discovery providers (Apify, Agent-Reach) and the downstream analytical, ranking, and multimodal intelligence engines.

## 2. Core Protocol Principles
1. **Production Invariance**: The default provider remains `apify/instagram-search-scraper`. Downstream ranking (reach sorting), relevance filtering (semantic thresholding), competitor intelligence (observable recurring patterns), and recommendation synthesis remain 100% unchanged.
2. **Provider Isolation**: Cache keys, query states, and adapters are strictly partitioned. Apify cached responses never satisfy Agent-Reach queries, and vice versa.
3. **Reach & Metadata Integrity**: Reach represents genuine video views/plays as a positive integer, or `None` if missing. Never fabricate reach, never use `0` to denote missing reach, never use likes or comments as views, and never infer creator usernames or timestamps.
4. **Canonical Identity & Deduplication**:
   - Reel identity is resolved exclusively through shortcode/Reel ID and canonical `/reel/{shortcode}/` URLs (stripping query parameters).
   - Captions and creator usernames are never used as identities.
5. **Cross-Provider Provenance Merging**: When multiple providers return the same Reel, the record is consolidated without overwriting populated fields with nulls, tracking all `source_providers` and `source_queries`.

## 3. Query Execution States
- `SUCCESS`: Query executed and returned valid candidate records.
- `BLOCKED`: Instagram platform challenge, login gate, or rate limit detected. Unretried; immediately proceeds to subsequent queries.
- `EMPTY`: Query executed successfully with 0 results returned.
- `TIMEOUT`: Network socket or provider execution exceeded deadline.
- `ERROR`: HTTP 5xx, credential absence, or runtime exception.
