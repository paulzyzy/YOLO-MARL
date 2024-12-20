import numpy as np

def planning_function(processed_state):
    """
    Determines optimal tasks for each agent based on the current state.
    
    Args:
        processed_state: A tuple containing food location and level, agent position and level.

    Returns:
        dict: Optimal tasks for each agent ('No op','Target food 0','Target food 1','Pickup')
    """
    food_info, agents_info = processed_state
    llm_tasks = {}
    
    # Find available food items
    available_food = [f for f, info in food_info.items() if info is not None]
    
    if not available_food:
        return {agent: 'No op' for agent in agents_info}
    
    # Calculate total agent level
    total_agent_level = sum(level for _, level in agents_info.values())
    
    # Choose target food (prioritize food that requires both agents)
    target_food = max(available_food, key=lambda f: food_info[f][1])
    target_food_pos, target_food_level = food_info[target_food]
    
    for agent, (agent_pos, _) in agents_info.items():
        distance = np.linalg.norm(np.array(agent_pos) - np.array(target_food_pos))
        
        if distance <= 1:
            llm_tasks[agent] = 'Pickup'
        else:
            llm_tasks[agent] = f'Target food {int(target_food[-1])}'
    
    return llm_tasks

def compute_reward(processed_state, llm_actions, actions):
    """
    Calculate rewards based on the tasks assigned and their outcomes.
    
    Args:
        processed_state: returned from function process_state(state, p, f)
        llm_actions (dict): dictionary of list of integers which means the suggest actions from llm for each agent.
        actions (dict): dictionary of a integer action that actually perform by each agent.
        
    Returns:
        reward: Dict containing rewards for each agent.
    """
    food_info, agents_info = processed_state
    reward = {agent: 0 for agent in agents_info}
    
    # Check if all agents are targeting the same food
    target_foods = set()
    for agent, action_list in llm_actions.items():
        if action_list and action_list[0] in [1, 2, 3, 4]:  # Movement actions
            target_foods.add(tuple(food_info[f'food_{action_list[0]-1}'][0]))
    
    coordinated_targeting = len(target_foods) == 1
    
    for agent in agents_info:
        # Reward for following LLM suggestion
        if actions[agent] in llm_actions[agent]:
            reward[agent] += 0.5
        
        # Reward for coordinated targeting
        if coordinated_targeting:
            reward[agent] += 1
        
        # Reward for pickup attempt
        if actions[agent] == 5:  # Pickup action
            agent_pos = agents_info[agent][0]
            nearby_food = [f for f, info in food_info.items() if info is not None and np.linalg.norm(np.array(agent_pos) - np.array(info[0])) <= 1]
            
            if nearby_food:
                food = nearby_food[0]
                food_level = food_info[food][1]
                total_agent_level = sum(level for _, level in agents_info.values())
                
                if total_agent_level >= food_level:
                    reward[agent] += 10  # Successful pickup
                else:
                    reward[agent] -= 1  # Failed pickup attempt
            else:
                reward[agent] -= 2  # Pickup attempt with no nearby food
    
    return reward