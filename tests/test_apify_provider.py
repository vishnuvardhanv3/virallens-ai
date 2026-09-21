"""Tests for Apify provider candidate normalization and limits."""
import apify_client
from instagram.providers.apify import ApifyProvider


def test_apify_client_import_resolution():
    # Must resolve to site-packages, never local shadow directory
    mod_path = apify_client.__file__
    assert "site-packages" in mod_path.lower()
    assert "viralens" not in mod_path.lower()


def test_candidate_normalization_and_filtering():
    provider = ApifyProvider()
    raw_items = [
        # 1. Valid GraphVideo post
        {
            "id": "12345",
            "shortcode": "C2_vRNYhykr",
            "url": "https://www.instagram.com/p/C2_vRNYhykr/",
            "type": "GraphVideo",
            "likes": 5000,
            "comments": 200,
            "videoViewCount": 75000,
            "videoUrl": "https://cdn.instagram.com/reel1.mp4",
            "caption": "Spiderman cinematic movie edit #superhero",
            "ownerUsername": "spidey_fan",
        },
        # 2. Valid Reel record with productType='clips'
        {
            "id": "67890",
            "shortCode": "DKeoUm9yiBV",
            "permalink": "https://www.instagram.com/reel/DKeoUm9yiBV/",
            "productType": "clips",
            "likesCount": 12000,
            "videoPlayCount": 200000,
            "media_url": "https://cdn.instagram.com/reel2.mp4",
            "caption": "Batman vs Joker scene edit",
            "user": {"username": "batman_edits"},
        },
        # 3. Invalid: Image-only post (should be rejected)
        {
            "id": "99999",
            "shortcode": "IMAGEONLY01",
            "url": "https://www.instagram.com/p/IMAGEONLY01/",
            "type": "image",
            "productType": "feed",
            "isVideo": False,
            "caption": "Just a photo",
        },
        # 4. Invalid: Hashtag directory record (should be rejected)
        {
            "searchType": "hashtag",
            "name": "superhero",
            "postsCount": 5000000,
        },
        # 5. Invalid: User profile page (should be rejected)
        {
            "isProfile": True,
            "username": "superhero_page",
            "profilePicUrl": "https://cdn.instagram.com/pic.jpg",
        },
    ]

    candidates = provider._normalize_candidates(raw_items)

    # Only the 2 genuine video candidates should be retained
    assert len(candidates) == 2

    c1 = candidates[0]
    assert c1["shortcode"] == "C2_vRNYhykr"
    # Canonical /p/ -> /reel/ URL normalization
    assert c1["url"] == "https://www.instagram.com/reel/C2_vRNYhykr/"
    assert c1["likes"] == 5000
    assert c1["comments"] == 200
    assert c1["views"] == 75000
    assert c1["username"] == "spidey_fan"

    c2 = candidates[1]
    assert c2["shortcode"] == "DKeoUm9yiBV"
    assert c2["url"] == "https://www.instagram.com/reel/DKeoUm9yiBV/"
    assert c2["likes"] == 12000
    assert c2["views"] == 200000
    assert c2["username"] == "batman_edits"


def test_candidate_deduplication():
    provider = ApifyProvider()
    duplicate_items = [
        {"id": "SAME1", "shortcode": "DUPLICATE01", "type": "video", "media_url": "http://vid1.mp4", "likes": 10},
        {"id": "SAME1", "shortcode": "DUPLICATE01", "type": "video", "media_url": "http://vid1.mp4", "likes": 10},
    ]
    candidates = provider._normalize_candidates(duplicate_items)
    assert len(candidates) == 1
