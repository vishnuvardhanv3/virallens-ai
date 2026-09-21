"""Tests for MasterAgent pipeline orchestration."""
from unittest.mock import patch, MagicMock
from core.master_agent import MasterAgent


def test_master_agent_pipeline_execution(tmp_path):
    # Create a dummy video file
    dummy_video = tmp_path / "test_reel.mp4"
    dummy_video.write_bytes(b"dummy video content")

    agent = MasterAgent()

    # Mock TwelveLabs analyze_video
    mock_tl_analysis = {
        "success": True,
        "analysis_status": "success",
        "analysis": {
            "niche": "superhero",
            "topic": "spiderman fight",
            "primary_entity": "spiderman",
            "format": "cinematic edit",
            "hook": {"type": "visual", "strength": 8, "first_3_seconds": "Spiderman leap"},
            "visual": {"visual_impact": 8, "scene_count": 5},
            "discovery_keywords": ["spiderman edit", "superhero marvel"],
        },
    }

    with patch.object(agent.analyzer, "analyze_video", return_value=mock_tl_analysis),          patch.object(agent.router, "search_with_diagnostics") as mock_search,          patch("core.master_agent.acquire") as mock_acquire,          patch("core.master_agent.analyze_video", return_value={"duration_sec": 15.0, "fps": 30.0, "frame_count": 450, "scene_change_count": 4}),          patch("core.master_agent.analyze_audio", return_value={"has_audio": True, "tempo": 120.0, "rms_mean": 0.05}),          patch("core.master_agent.transcribe", return_value={"has_speech": True, "speech_text": "I'm Spiderman"}),          patch("core.master_agent.extract_ocr", return_value={"ocr_available": True, "ocr_text": "SPIDERMAN"}):

        # Mock Apify search returning candidates
        mock_candidates = [
            {
                "id": "c_spidey1",
                "shortcode": "c_spidey1",
                "url": "https://www.instagram.com/reel/c_spidey1/",
                "caption": "Spiderman fight scene edit #superhero",
                "source_query": "superhero spiderman cinematic edit",
                "media_url": "http://example.com/c1.mp4",
                "views": 100000,
                "likes": 5000,
            },
            {
                "id": "c_spidey2",
                "shortcode": "c_spidey2",
                "url": "https://www.instagram.com/reel/c_spidey2/",
                "caption": "Spiderman swinging across New York",
                "source_query": "superhero spiderman edit",
                "media_url": "http://example.com/c2.mp4",
                "views": 80000,
                "likes": 4000,
            }
        ]
        mock_search.return_value = (mock_candidates, {"raw_candidates": 2, "status": "SUCCESS"})

        # Mock media acquire
        mock_acquire.side_effect = lambda cand, dest: {
            "media_status": "available",
            "video_download_status": "DOWNLOADED",
            "local_path": str(dummy_video),
            "cached": True,
            "error": "",
        }

        # Run pipeline
        res = agent.run(dummy_video)

        assert res["your_reel"]["niche"] == "superhero"
        assert res["your_reel"]["primary_entity"] == "spiderman"
        assert len(res["queries"]) >= 3
        assert len(res["verified_competitors"]) == 2
        assert res["report_json_path"] is not None
        assert res["report_md_path"] is not None
