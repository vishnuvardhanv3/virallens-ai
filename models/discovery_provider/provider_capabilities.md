# Provider Capabilities Matrix

## 1. Ground Truth Capability Audit
Capabilities in ViralLens AI reflect observed, verified engineering reality, never aspirational assumptions or promotional claims.

| Capability Field | Apify (`apify/instagram-search-scraper`) | Agent-Reach / OpenCLI (Experimental) |
| :--- | :--- | :--- |
| **Search by Keyword / Niche** | ✅ Supported (`searchType="popular"`) | ⛔ Requires local CLI + Session |
| **Metadata Extraction** | ✅ Supported (JSON Dataset) | ⛔ Requires local CLI + Session |
| **Direct Reach (Plays / Views)** | ✅ Supported (`videoPlayCount`, `views`) | ⛔ Unavailable in current environment |
| **Direct Media URL** | ✅ Supported (`videoUrl`, CDN link) | ⛔ False (Does not expose direct video URLs) |
| **Pagination** | ✅ Supported (`searchLimit` parameter) | ⛔ Fixed single-pass query |
| **Requires Authenticated Session** | ❌ No (Handled by Apify actor proxies) | ✅ Yes (Requires desktop cookie/session) |
| **Profile Lookup** | ✅ Supported | ⛔ Unavailable in current environment |
| **Operational Status** | **Production Ready (Default)** | **Experimental / Unavailable on Host** |

## 2. Safety & Platform Compliance
Neither provider utilizes browser automation workarounds, unauthorized credential harvesting, or unvetted scraping libraries:
- ❌ No Selenium
- ❌ No Playwright
- ❌ No Instaloader
- ❌ No Instagrapi
