def compute_reward(processed_state, actions):
    """
    Calculate rewards based on the tasks assigned and their outcomes.
    
    Args:
        processed_state: returned from function process_state(state, p, f)
        actions (dict): dictionary of a integer action that actually perform by each agent. E.g. {"agent_0": 2, "agent_1": 4, ...}

    Returns:
        reward: Dict containing rewards for each agent. For example: {'agent_0': reward1, 'agent_1', reward2, ...}
    """
    food_info, agents_info = processed_state
    reward = {agent_id: 0 for agent_id in agents_info.keys()}
    
    # Reward for picking up food
    pickup_agents = [agent_id for agent_id, action in actions.items() if action == 5]
    if len(pickup_agents) == len(agents_info):  # All agents attempting pickup
        food_positions = [food[0] for food in food_info.values() if food is not None]
        if food_positions and all(any(abs(agents_info[agent_id][0][0] - food_pos[0]) + abs(agents_info[agent_id][0][1] - food_pos[1]) <= 1 for food_pos in food_positions) for agent_id in pickup_agents):
            total_agent_level = sum(agents_info[agent_id][1] for agent_id in pickup_agents)
            food_level = max(food[1] for food in food_info.values() if food is not None)
            if total_agent_level >= food_level:
                for agent_id in pickup_agents:
                    reward[agent_id] += 100  # Higher reward for successful coordinated pickup
    
    # Reward for moving towards food and staying close to other agents
    for agent_id, action in actions.items():
        if action in [1, 2, 3, 4]:  # Moving actions
            agent_pos = agents_info[agent_id][0]
            closest_food = min((food for food in food_info.values() if food is not None), 
                               key=lambda f: abs(agent_pos[0] - f[0][0]) + abs(agent_pos[1] - f[0][1]), 
                               default=None)
            if closest_food:
                old_distance = abs(agent_pos[0] - closest_food[0][0]) + abs(agent_pos[1] - closest_food[0][1])
                new_pos = list(agent_pos)
                if action == 1: new_pos[0] -= 1
                elif action == 2: new_pos[0] += 1
                elif action == 3: new_pos[1] -= 1
                elif action == 4: new_pos[1] += 1
                new_distance = abs(new_pos[0] - closest_food[0][0]) + abs(new_pos[1] - closest_food[0][1])
                if new_distance < old_distance:
                    reward[agent_id] += 5  # Increased reward for moving closer to food
            
            # Reward for staying close to other agents
            other_agents = [a for a in agents_info.keys() if a != agent_id]
            for other_agent in other_agents:
                other_pos = agents_info[other_agent][0]
                old_agent_distance = abs(agent_pos[0] - other_pos[0]) + abs(agent_pos[1] - other_pos[1])
                new_agent_distance = abs(new_pos[0] - other_pos[0]) + abs(new_pos[1] - other_pos[1])
                if new_agent_distance < old_agent_distance:
                    reward[agent_id] += 3  # Increased reward for decreasing distance to other agents
    
    # Penalty for no-op when food is available
    for agent_id, action in actions.items():
        if action == 0 and any(food is not None for food in food_info.values()):
            reward[agent_id] -= 5  # Increased penalty for no-op when food is available
    
    return reward
