import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import os
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Add workspace to path to find humanoid_walker module
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    
    from humanoid_walker.humanoid_walker import HumanoidWalker
    return HumanoidWalker, mo, np, os, pd, plt, sys, time


@app.cell
def _(mo):
    mo.md(
        r"""
        # Humanoid Walking Simulation (Marimo Edition)

        This is a Marimo notebook that runs the humanoid walking simulation. 
        It demonstrates how to use `anywidget` with Marimo for interactive simulation and learning.
        """
    )
    return


@app.cell
def _(HumanoidWalker, mo):
    walker = HumanoidWalker()
    
    # Create UI controls
    start_btn = mo.ui.button(label="Start Learning Loop")
    stop_btn = mo.ui.button(label="Stop")
    reset_btn = mo.ui.button(label="Reset Simulation")
    
    mo.vstack([
        mo.md("### Simulation View"),
        walker,
        mo.hstack([start_btn, stop_btn, reset_btn], justify="center")
    ])
    return reset_btn, start_btn, stop_btn, walker


@app.cell
def _(reset_btn, walker):
    if reset_btn.value:
        walker.reset_simulation()
    return


@app.cell
def _(mo):
    # State for the learning loop
    get_state, set_state = mo.state({
        "running": False,
        "episode": 0,
        "best_reward": -float('inf'),
        "best_weights": None,
        "logs": []
    })
    return get_state, set_state


@app.cell
def _(get_state, set_state, start_btn, stop_btn):
    # Update running state based on buttons
    if start_btn.value:
        set_state({**get_state(), "running": True})
    
    if stop_btn.value:
        set_state({**get_state(), "running": False})
    return


@app.cell
def _(get_state, mo):
    # Display current status
    state = get_state()
    status_text = "Running..." if state["running"] else "Stopped"
    mo.md(f"""
    **Status**: {status_text}
    
    **Episode**: {state["episode"]}
    **Best Reward**: {state["best_reward"]:.2f}
    """)
    return state, status_text


@app.cell
def _(get_state, set_state, time, walker, np):
    # The Learning Loop (runs continuously if running is True)
    
    # We need to import the logic functions here or define them
    def get_action(weights, state_vec):
        action_vec = np.tanh(np.dot(weights, state_vec)) * 100 
        return {
            'left_hip': action_vec[0],
            'left_knee': action_vec[1],
            'right_hip': action_vec[2],
            'right_knee': action_vec[3]
        }

    # This cell needs to re-run itself or be driven by a periodic trigger if we want continuous updates in Marimo without blocking everything.
    # However, Marimo cells are reactive. A long running loop blocks the UI.
    # We should use a background thread or `mo.ui.refresh`?
    # For now, let's just do ONE step of the learning process when "running" is true, 
    # and use `mo.ui.refresh` to trigger the next step? 
    # Actually, a better pattern for simulations in Marimo is often `mo.status` or a background thread that updates a shared object,
    # but anywidget updates are already async.
    
    # Let's run a single episode if running is True, then update state which triggers re-run?
    # No, modifying state triggers re-run of dependent cells, so we can loop that way!
    
    current_state = get_state()
    
    if current_state["running"]:
        # Run one episode
        # Initialize weights if needed
        best_weights = current_state["best_weights"]
        if best_weights is None:
            best_weights = np.random.randn(4, 6) * 0.1
        
        # Mutate
        candidate_weights = best_weights + np.random.randn(4, 6) * 0.5
        
        # Run episode
        walker.reset_simulation()
        time.sleep(0.2) # Short pause for reset
        
        initial_x = walker.state.get('torso_x', 200)
        
        # Run for fixed steps (blocking this cell, but that's ok for sequential episodes)
        # 100 steps @ 0.05s = 5 seconds per episode
        for _ in range(50): 
            s = walker.state
            if not s:
                time.sleep(0.05)
                continue
            
            state_vec = np.array([
                s.get('torso_angle', 0),
                s.get('left_thigh_angle', 0),
                s.get('left_calf_angle', 0),
                s.get('right_thigh_angle', 0),
                s.get('right_calf_angle', 0),
                1.0 
            ])
            
            action = get_action(candidate_weights, state_vec)
            walker.apply_action(action)
            time.sleep(0.05)
            
        final_x = walker.state.get('torso_x', initial_x)
        reward = final_x - initial_x
        
        # Update State
        new_best_reward = current_state["best_reward"]
        new_best_weights = best_weights
        
        if reward > new_best_reward:
            new_best_reward = reward
            new_best_weights = candidate_weights
            
        # Log
        new_logs = current_state["logs"] + [{"episode": current_state["episode"], "reward": reward}]
        
        # Update state (this effectively schedules the next run if we self-reference carefully, 
        # but pure recursion might kill it. 
        # Actually, let's just use `mo.ui.refresh` or similar if we want a loop.
        # But simply setting state here will NOT cause infinite loop unless this cell DEPENDS on state 
        # AND sets it. It depends on `get_state`.
        
        # To make it loop, we need to trigger a re-run. 
        # We can use a button click to trigger "Next Episode" manually, or use `mo.refresh`.
        # For auto-play, we can use `mo.ui.refresh` with interval.
        
        # Let's just update the state. The user has to click "Start" again? 
        # No, let's use a refresh interval.
        
        set_state({
            **current_state,
            "episode": current_state["episode"] + 1,
            "best_reward": new_best_reward,
            "best_weights": new_best_weights,
            "logs": new_logs
        })

    return get_action,


@app.cell
def _(mo, get_state, set_state):
    # Auto-runner
    # If running is true, we want to trigger the loop again.
    state = get_state()
    if state["running"]:
        # Re-run every 100ms (after the previous cell finished)
        mo.ui.refresh(default_interval="100ms") 
    return


@app.cell
def _(get_state, mo, pd):
    # Plotting results
    state = get_state()
    if state["logs"]:
        df = pd.DataFrame(state["logs"])
        chart = mo.ui.table(df) # or plot
        
        # Simple plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["episode"], df["reward"])
        ax.set_title("Reward over Episodes")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Distance Traveled")
        
        mo.vstack([chart, mo.as_html(fig)])
    return


if __name__ == "__main__":
    app.run()
