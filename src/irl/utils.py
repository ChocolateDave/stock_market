from typing import Mapping, Sequence, Any
from src.memory.base_buffer import Path

def load_expert_behavior(
        expert_behavior_dict: Mapping[str, Any]
    ) -> Sequence[Path]:
    obs = expert_behavior_dict['observations']
    acs = expert_behavior_dict['actions']
    next_obs = expert_behavior_dict['next_observations']
    rews = expert_behavior_dict['rewards']
    dones = expert_behavior_dict['dones']
    
    start_idx = 0
    paths = []
    for i in range(len(dones)):
        if dones[i] == True:
            path = Path(
                obs[start_idx:i + 1],
                acs[start_idx:i + 1],
                next_obs[start_idx:i + 1],
                rews[start_idx:i + 1],
                dones[start_idx:i + 1].astype(int)
            )
            paths.append(path)
            start_idx = i + 1
    return paths
