# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["marimo", "anywidget", "traitlets", "numpy"]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import time
    import random
    from pathlib import Path
    import anywidget
    import traitlets
    import marimo as mo
    return Path, anywidget, mo, np, random, time, traitlets


@app.cell
def _(Path, anywidget, traitlets):
    _ASSET_DIR = Path(__file__).parent
    _JS_SOURCE = (_ASSET_DIR / "maze_widget.js").read_text()
    _CSS_SOURCE = (_ASSET_DIR / "maze_widget.css").read_text()

    class MazeWidget(anywidget.AnyWidget):
        _esm = _JS_SOURCE
        _css = _CSS_SOURCE

        # 0=Empty, 1=Wall, 2=Start, 3=Goal
        maze_layout = traitlets.List(traitlets.List(traitlets.Int())).tag(sync=True)
        agent_position = traitlets.List(traitlets.Int()).tag(sync=True)
        # Grid of directions (0=Up, 1=Right, 2=Down, 3=Left, -1=None)
        policy_grid = traitlets.List(traitlets.List(traitlets.Int())).tag(sync=True)
        # Full Q-table: List of List of List (Rows x Cols x 4)
        q_values = traitlets.List(traitlets.List(traitlets.List(traitlets.Float()))).tag(sync=True)
        show_q_values = traitlets.Bool(default_value=False).tag(sync=True)
        active_tool = traitlets.Int(default_value=0).tag(sync=True)

        def __init__(self, layout, start_pos):
            super().__init__()
            self.maze_layout = layout
            self.agent_position = start_pos
            self.policy_grid = [[-1] * len(row) for row in layout]
            self.q_values = []

    return (MazeWidget,)


@app.cell
def _(np, random):
    class QLearningAgent:
        def __init__(self, rows, cols, actions=4, alpha=0.1, gamma=0.9, epsilon=0.1):
            self.q_table = np.zeros((rows, cols, actions))
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.actions = actions # 0: Up, 1: Right, 2: Down, 3: Left
            self.rng = np.random.default_rng()

        def choose_action(self, state):
            if self.rng.random() < self.epsilon:
                return self.rng.integers(0, self.actions)
            else:
                return self.get_best_action(state)

        def get_best_action(self, state):
            # Break ties randomly
            values = self.q_table[state[0], state[1]]
            max_val = np.max(values)
            candidates = np.where(values == max_val)[0]
            return self.rng.choice(candidates)

        def learn(self, state, action, reward, next_state):
            old_value = self.q_table[state[0], state[1], action]
            next_max = np.max(self.q_table[next_state[0], next_state[1]])
            
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            self.q_table[state[0], state[1], action] = new_value

        def get_policy_grid(self):
            rows, cols, _ = self.q_table.shape
            policy = [[-1] * cols for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    # Only show policy if visited/learned (non-zero) or just best
                    if np.any(self.q_table[r, c] != 0):
                        policy[r][c] = int(self.get_best_action((r, c)))
            return policy

    return (QLearningAgent,)


@app.cell
def _(MazeWidget, QLearningAgent, mo, np, time):
    # Default Maze Layout
    # 0=Empty, 1=Wall, 2=Start, 3=Goal, 4=Danger
    DEFAULT_MAZE = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 4, 0, 1, 0, 1], # Added a Danger spot (4)
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 3, 1],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    START_POS = [1, 1]
    
    # Initialize Widget
    maze_widget = MazeWidget(layout=DEFAULT_MAZE, start_pos=START_POS)
    
    # UI Controls
    epsilon_slider = mo.ui.slider(0.0, 1.0, step=0.01, value=0.1, label="Epsilon (Exploration)")
    alpha_slider = mo.ui.slider(0.0, 1.0, step=0.01, value=0.1, label="Alpha (Learning Rate)")
    gamma_slider = mo.ui.slider(0.0, 1.0, step=0.01, value=0.9, label="Gamma (Discount)")
    speed_slider = mo.ui.slider(0.01, 1.0, step=0.01, value=0.1, label="Step Delay (s)")
    
    show_q_toggle = mo.ui.switch(label="Show Q-Values", value=False)
    
    tool_selector = mo.ui.radio(
        options={"Empty": 0, "Wall": 1, "Start": 2, "Goal": 3, "Danger": 4},
        value=0,
        label="Edit Tool"
    )

    start_btn = mo.ui.run_button(label="Train Episode")
    reset_btn = mo.ui.button(label="Reset Q-Table")

    # Global Agent State
    agent_ref = {"obj": QLearningAgent(len(DEFAULT_MAZE), len(DEFAULT_MAZE[0]))}

    def reset_agent():
        # Re-read dimensions from current widget state in case it changed (though resizing isn't supported yet, but layout content is)
        current_layout = maze_widget.maze_layout
        rows = len(current_layout)
        cols = len(current_layout[0])
        
        agent_ref["obj"] = QLearningAgent(
            rows, 
            cols,
            alpha=alpha_slider.value,
            gamma=gamma_slider.value,
            epsilon=epsilon_slider.value
        )
        maze_widget.policy_grid = [[-1]*cols]*rows # Clear arrows
        maze_widget.q_values = []
        
        # Reset position to current start
        start_found = False
        for r in range(rows):
            for c in range(cols):
                if current_layout[r][c] == 2:
                    maze_widget.agent_position = [r, c]
                    start_found = True
                    break
            if start_found: break

    def run_episode():
        agent = agent_ref["obj"]
        # Update params
        agent.alpha = alpha_slider.value
        agent.gamma = gamma_slider.value
        agent.epsilon = epsilon_slider.value
        
        # Find start and goal from current layout
        layout = maze_widget.maze_layout
        start_pos = None
        goal_pos = None
        
        for r, row in enumerate(layout):
            for c, cell in enumerate(row):
                if cell == 2: start_pos = [r, c]
                elif cell == 3: goal_pos = [r, c]
        
        if not start_pos: return # No start
        
        state = list(start_pos)
        maze_widget.agent_position = state
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            action = agent.choose_action(tuple(state))
            
            # Execute action
            # 0: Up, 1: Right, 2: Down, 3: Left
            next_r, next_c = state
            if action == 0: next_r -= 1
            elif action == 1: next_c += 1
            elif action == 2: next_r += 1
            elif action == 3: next_c -= 1
            
            # Check bounds
            if next_r < 0 or next_r >= len(layout) or next_c < 0 or next_c >= len(layout[0]):
                next_state = state
                reward = -1 # Wall penalty
            else:
                cell_type = layout[next_r][next_c]
                
                if cell_type == 1: # Wall
                    next_state = state
                    reward = -1
                elif cell_type == 3: # Goal
                    next_state = [next_r, next_c]
                    reward = 100
                    done = True
                elif cell_type == 4: # Danger
                    next_state = [next_r, next_c]
                    reward = -100
                    done = True # Lose life / End episode
                else: # Empty or Start (treated as empty)
                    next_state = [next_r, next_c]
                    reward = -0.1 # Step cost
            
            agent.learn(tuple(state), action, reward, tuple(next_state))
            state = next_state
            
            # Update UI
            maze_widget.agent_position = state
            if steps % 5 == 0: # Update occasionally to save overhead
                 maze_widget.q_values = agent.q_table.tolist()
            
            steps += 1
            time.sleep(speed_slider.value)
        
        # Update policy view after episode
        maze_widget.policy_grid = agent.get_policy_grid()
        maze_widget.q_values = agent.q_table.tolist()

    return (
        DEFAULT_MAZE,
        START_POS,
        agent_ref,
        alpha_slider,
        epsilon_slider,
        gamma_slider,
        maze_widget,
        reset_agent,
        reset_btn,
        run_episode,
        show_q_toggle,
        speed_slider,
        start_btn,
        tool_selector,
    )


@app.cell
def _(mo):
    mo.md(r"""# Q-Learning Maze Simulation""")
    return


@app.cell
def _(
    alpha_slider,
    epsilon_slider,
    gamma_slider,
    maze_widget,
    mo,
    reset_agent,
    reset_btn,
    run_episode,
    show_q_toggle,
    speed_slider,
    start_btn,
    tool_selector,
):
    # UI Layout
    
    # Handle Reset
    if reset_btn.value:
        reset_agent()

    # Handle Train
    if start_btn.value:
        run_episode()

    # Sync toggle
    maze_widget.show_q_values = show_q_toggle.value
    # Sync tool
    maze_widget.active_tool = tool_selector.value

    mo.vstack([
        mo.hstack([maze_widget, mo.vstack([
            mo.md("### Controls"),
            start_btn, 
            reset_btn,
            mo.md("---"),
            epsilon_slider,
            alpha_slider,
            gamma_slider,
            speed_slider,
            show_q_toggle,
            mo.md("---"),
            tool_selector
        ])], align="start", gap="2rem"),
    ])
    return


if __name__ == "__main__":
    app.run()
