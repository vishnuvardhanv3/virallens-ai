from pathlib import Path
from services.media_service import validate_video_file, acquire


def test_validate_nonexistent_video():
    res = validate_video_file('nonexistent_video_12345.mp4')
    assert res['valid'] is False
    assert 'exist' in res['error'].lower()


def test_acquire_failure_sets_download_failed(tmp_path):
    cand = {
        'id': 'test_fail_01',
        'shortcode': 'test_fail_01',
        'media_url': '',
    }
    res = acquire(cand, tmp_path)
    assert res['video_download_status'] == 'DOWNLOAD_FAILED'
    assert res['media_status'] == 'media_unavailable'
