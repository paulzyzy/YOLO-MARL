import numpy as np
def get_relative_position(agent_pos, target_pos):
    '''
    Helper function to calculate the relative position of target to the agent.
    Returns a tuple (dx, dy) where:
    dx: Difference in the x-coordinate.
    dy: Difference in the y-coordinate.
    '''
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    return dx, dy

def get_rware_actions(direction, relative_pos, can_move_forward):
    '''
    Helper function to determine the movement actions based on the relative position.
    Returns a list of movement actions [move1, move2].
    1: Up (X-)
    2: Down (X+)
    3: Left (Y-)
    4: Right (Y+)
    '''
    dx, dy = relative_pos

    actions = []
    if dx == 0 and dy == 0:
        return [4]
    if direction == 'up': 
        if can_move_forward and dy < 0:
            actions.append(1)
        if dx < 0:
            actions.append(2)
        if dx > 0:
            actions.append(3)
        if dx == 0:
            if dy > 0:
                actions.extend([2, 3])
            elif dy < 0 and not can_move_forward:
                actions.extend([2, 3])

    elif direction == 'down':
        if can_move_forward and dy > 0:
            actions.append(1)
        if dx < 0:
            actions.append(3)
        if dx > 0:
            actions.append(2)
        if dx == 0:
            if dy < 0:
                actions.extend([2, 3])
            elif dy > 0 and not can_move_forward:
                actions.extend([2, 3])

    elif direction == 'left':
        if can_move_forward and dx < 0:
            actions.append(1)
        if dy < 0:
            actions.append(3)
        if dy > 0:
            actions.append(2)
        if dy == 0:
            if dx > 0:
                actions.extend([2, 3])
            elif dx < 0 and not can_move_forward:
                actions.extend([2, 3])
        
    elif direction == 'right':
        if can_move_forward and dx > 0:
            actions.append(1)
        if dy < 0:
            actions.append(2)
        if dy > 0:
            actions.append(3)
        if dy == 0:
            if dx < 0:
                actions.extend([2, 3])
            elif dx > 0 and not can_move_forward:
                actions.extend([2, 3])

    return actions

def rware_task_to_actions(task, processed_obs):
    action = {agent: [] for agent in task.keys()}
    agent_infos = processed_obs['agent_infos']

    for agent in task.keys():
        agent_task = task[agent]
        agent_index = int(agent.split('_')[1])  # Extract the index from the agent string
        agent_info = agent_infos[agent_index]
        agent_pos = agent_info['location']
        agent_direction = agent_info['direction']
        agent_can_move_forward = agent_info['can_move_forward']
        agent_can_place_shelf = agent_info['can_place_shelf']

        if agent_task == "random explore":
            if agent_info['can_move_forward']:
                action[agent].extend([1, 2, 3])
            else:
                action[agent].extend([2, 3])

        elif agent_task == "empty shelf":
            empty_shelf_pos = processed_obs['empty_shelves_pos']
            if not empty_shelf_pos:
                if agent_info['can_move_forward']:
                    action[agent].extend([1, 2, 3])
                else:
                    action[agent].extend([2, 3])
            else:
                target = min(empty_shelf_pos, key=lambda pos: sum(abs(a - b) for a, b in zip(pos, agent_pos)))
                relative_pos = get_relative_position(agent_pos, target)
                action[agent].extend(get_rware_actions(agent_direction, relative_pos, agent_can_move_forward))
        
        elif agent_task == "workstation":
            workstation_locations = processed_obs['workstation location']
            closest_workstation = min(workstation_locations, 
                    key=lambda pos: sum(abs(a - b) for a, b in zip(pos, agent_pos)))
            relative_pos = get_relative_position(agent_pos, closest_workstation)
            action[agent].extend(get_rware_actions(agent_direction, relative_pos, agent_can_move_forward))
            # workstation_pos_1 = processed_obs['workstation location'][0]
            # workstation_pos_2 = processed_obs['workstation location'][1]
            # relative_pos_1 = get_relative_position(agent_pos, workstation_pos_1)
            # relative_pos_2 = get_relative_position(agent_pos, workstation_pos_2)
            # action[agent].extend(get_rware_actions(agent_direction, relative_pos_1, agent_can_move_forward))
            # action[agent].extend(get_rware_actions(agent_direction, relative_pos_2, agent_can_move_forward))

        elif agent_task == "return":
            return_locations = processed_obs['return location']
            if not processed_obs['return location']:
                if agent_info['can_move_forward']:
                    action[agent].extend([1, 2, 3])
                else:
                    action[agent].extend([2, 3])
            else:
                closest_return = min(return_locations, 
                        key=lambda pos: sum(abs(a - b) for a, b in zip(pos, agent_pos)))
                relative_pos = get_relative_position(agent_pos, closest_return)
                action[agent].extend(get_rware_actions(agent_direction, relative_pos, agent_can_move_forward))
                # for return_pos in processed_obs['return location']: 
                #     relative_pos = get_relative_position(agent_pos, return_pos)
                #     action[agent].extend(get_rware_actions(agent_direction, relative_pos, agent_can_move_forward))
        
    return action