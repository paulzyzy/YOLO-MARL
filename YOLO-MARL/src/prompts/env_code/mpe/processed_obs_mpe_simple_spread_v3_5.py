import numpy as np
def process_state(observations, N=5):
    '''
    Param:
        observations:
            List of NumPy arrays, one per agent.
            Each array represents the observation for an agent:
            [self_vel (2,), self_pos (2,), landmark_rel_positions (N*2,), other_agent_rel_positions ((N-1)*2,), communication]

    Return:
        obs:
            Dictionary with agent IDs as keys ('agent_0', 'agent_1', ...).
            Each value is a list containing:
                - Landmark relative positions: N arrays of shape (2,)
                - Other agents' relative positions: (N-1) arrays of shape (2,)
    '''
    obs = {}
    num_agents = len(observations)

    for idx, agent_obs in enumerate(observations):
        agent_id = f'agent_{idx}'
        obs[agent_id] = []
        
        # Extract landmark relative positions
        for i in range(N):
            start = 4 + 2 * i
            end = start + 2
            land_2_a = agent_obs[start:end]
            obs[agent_id].append(land_2_a)
        
        # Extract other agents' relative positions
        for i in range(num_agents - 1):
            start = 4 + 2 * N + 2 * i
            end = start + 2
            other_agent_2_a = agent_obs[start:end]
            obs[agent_id].append(other_agent_2_a)

    return obs