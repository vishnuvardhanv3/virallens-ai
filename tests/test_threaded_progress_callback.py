"""
Test progress callback thread-safety.

Validates that MasterAgent._log and progress callbacks remain resilient
when called from concurrent worker threads during competitor processing.
"""
import concurrent.futures
import threading
import pytest
from core.master_agent import MasterAgent


def test_master_agent_log_thread_safety_with_failing_callback():
    """Verify that exceptions inside progress_callback do not escape _log or crash threads."""
    def failing_callback(msg: str):
        # Emulate Streamlit thread AttributeError: st.session_state has no attribute 'live_logs'
        raise AttributeError("st.session_state has no attribute 'live_logs'")

    agent = MasterAgent(progress_callback=failing_callback)

    # Calling _log from a worker thread must not raise
    def worker_task(idx: int):
        agent._log(f"Competitor {idx}/10: worker progress message")
        return idx * 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker_task, range(5)))

    assert results == [0, 2, 4, 6, 8]


def test_master_agent_log_thread_safety_with_thread_safe_collector():
    """Verify that a thread-safe collector receives all messages from worker threads."""
    captured = []
    lock = threading.Lock()

    def safe_callback(msg: str):
        with lock:
            captured.append(msg)

    agent = MasterAgent(progress_callback=safe_callback)

    def worker_task(idx: int):
        agent._log(f"Competitor {idx}: working")
        return idx

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker_task, range(4)))

    assert len(results) == 4
    assert len(captured) == 4
    assert any("Competitor 0" in m for m in captured)
    assert any("Competitor 3" in m for m in captured)
