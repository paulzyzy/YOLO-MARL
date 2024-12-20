import numpy as np
def process_state(observations, N=2, boundary=(10, 11)):
    '''
    Param:
        observations:
            List of NumPy arrays, one per agent.
        N: 
            Number of agents in the environment.
        boundary:
            Tuple representing the boundary of the grid world.

        Return:
        A dictionary containing global information for each agent and the positions of empty shelves observed by any agent.
            agent_infos:
                A list of dictionaries, one per agent, containing:
                    - 'location': Current location of the agent.
                    - 'is_carrying_shelf': Whether the agent is carrying a shelf.
                    - 'direction': Current heading direction ('up', 'down', 'left', 'right').
                    - 'can_move_forward': Whether the agent can move forward.
                    - 'can_place_shelf': Whether the agent can place a shelf at the current location. If False, the agent is at the path. If True, only means agent is not at the path, but not necessarily can place a shelf.
            empty_shelves_pos:
                List of positions of empty shelves observed by any agent. If the list is empty, then currently no empty shelf is around the agents.
    '''

    def boundary_check(pos, boundary):
        '''
        Check if the position is within the boundary.
        '''
        return pos[0] >= 0 and pos[0] < boundary[0] and pos[1] >= 0 and pos[1] < boundary[1]

    agent_infos = []
    empty_shelves_pos = []
    direction_offset = np.array([[-1, -1], [0, -1], [1, -1], 
                                 [-1, 0], [0, 0], [1, 0], 
                                 [-1, 1], [0, 1], [1, 1]])
    forward_offset = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
    for idx, agent_obs in enumerate(observations):
        agent_info = {}
        # Extract agent's self information
        self_info = agent_obs[:8]
        current_pos = self_info[:2]
        is_carrying_shelf = self_info[2]
        heading_direction = np.where(self_info[3:7] == 1)[0][0] # up/down/left/right
        can_place_shelf = self_info[7]
        # Extract agent's surrounding information
        surrounding_info = agent_obs[8:]
        start = 0
        can_move_forward = True
        for i in range(9):
            end = start + 7
            curr_square_info = surrounding_info[start:end]
            is_agent_exist = curr_square_info[0]
            other_agent_direction = curr_square_info[1:5]
            is_shelf_exist = curr_square_info[5]
            is_delivered_shelf = curr_square_info[6]
            # If one of the agent's observed square is occupied by shelf required to deliver, then add it to the list
            forward_pos = current_pos + direction_offset[i]
            if is_shelf_exist and is_delivered_shelf:
                empty_shelves_pos.append(forward_pos)
            # Agent cannot move forward if:
            # 1. It's out of boundary when move forward.
            # 2. There is another agent in front of it.
            # 3. There is a shelf in front of it and itself is carrying a shelf.
            if (not boundary_check(forward_pos, boundary) or is_agent_exist or (is_shelf_exist and is_carrying_shelf)) and \
                np.all(direction_offset[i] == forward_offset[heading_direction]):
                can_move_forward = False
            start = end
        agent_info['location'] = current_pos

        if heading_direction == 0:
            heading_direction = 'up'
        elif heading_direction == 1:
            heading_direction = 'down'
        elif heading_direction == 2:
            heading_direction = 'left'
        elif heading_direction == 3:
            heading_direction = 'right'

        agent_info['is_carrying_shelf'] = is_carrying_shelf
        agent_info['direction'] = heading_direction
        agent_info['can_move_forward'] = can_move_forward
        agent_info['can_place_shelf'] = not (can_place_shelf == 1)
        agent_infos.append(agent_info)

    unique_empty_shelves = []
    for pos in empty_shelves_pos:
        if not any(np.array_equal(pos, existing_pos) for existing_pos in unique_empty_shelves):
            unique_empty_shelves.append(pos)

    return {
        'agent_infos': agent_infos,
        'empty_shelves_pos': list(unique_empty_shelves)
        # 'empty_shelves_pos': empty_shelves_pos
    }

if __name__ == "__main__":
    # import pyglet
    # pyglet.window.Window()
    from rware_memory import rware_memory
    from generated_code import planning_function
    from task2action import rware_task_to_actions
    import gymnasium as gym
    import rware
    import os
    import imageio
    import time
    import cv2
    import glob
    import time
    # Directory to save the GIF
    # save_dir = "/home/paulzy/LLM-copilot-RL/LBF/src/prompts/env_code/rware/gifs"
    # os.makedirs(save_dir, exist_ok=True)

    # Create the environment with rgb_array rendering
    env = gym.make("rware:rware-tiny-2ag-v2", render_mode='rgb_array')
    memory = {
                'workstation location': [[4,10], [5,10]],
                'empty_shelves_pos': [],
                'return location': [],
                'status of carried shelf': [False, False],
                'is_carrying_shelf': [False, False]
            }
    frames = []
    obss, info = env.reset()
    pre_action = [[], []]
    # frame = env.render()
    # frames.append(frame)
    for i in range(500):
        print("obs", obss)
        processed_state = process_state(obss)
        print("pre_action", pre_action)

        processed_state, memory = rware_memory(processed_state, memory, pre_action)
        print("processed_state", processed_state)
        print("memory", memory) 
        
        llm_task = planning_function(processed_state)
        print("llm_task", llm_task)
        llm_actions = rware_task_to_actions(llm_task, processed_state)
        print("llm_actions", llm_actions)
        actions = env.action_space.sample()
        # print("actions", actions)
        obss, rewards, done, truncated, info = env.step(actions)
        print("rewards", rewards)
        # frame = env.render()
        # frames.append(frame)
        pre_action = actions
        if done:
            break
    env.close()

    # # Now save each frame as an image using cv2
    # for idx, frm in enumerate(frames):
    #     # env.render() returns RGB frames, cv2 uses BGR.
    #     frm_bgr = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(os.path.join(save_dir, f"frame_{idx:04d}.png"), frm_bgr)

    # # Now use imageio to create a GIF from the saved frames
    # gif_path = os.path.join(save_dir, f'episode_{int(time.time())}.gif')

    # images = []
    # for filename in sorted(glob.glob(os.path.join(save_dir, "frame_*.png"))):
    #     images.append(imageio.imread(filename))

    # imageio.mimsave(gif_path, images, fps=5)
    # print(f"GIF saved at {gif_path}")