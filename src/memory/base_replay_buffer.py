# =============================================================================
# @file   base_replay_buffer.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""The Base Replay Buffer data structure."""


class BaseReplayBuffer:

    def __init__(self, max_size: int = 10000) -> None:
        pass

    def add(self, path) -> None:
        return None

    def sample(self, path) -> None:
        return None
