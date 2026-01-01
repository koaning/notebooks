# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.18.0",
#     "gymnasium>=0.29.0",
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "matplotlib>=3.7.0",
#     "plotly>=5.14.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import gymnasium as gym
    from collections import deque
    import random
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import marimo as mo
    from typing import List, Tuple
    return (
        List,
        Tuple,
        deque,
        go,
        gym,
        make_subplots,
        matplotlib,
        mo,
        nn,
        np,
        optim,
        plt,
        random,
        torch,
    )


@app.cell
def _():
    # DQN Neural Network
    class DQN(nn.Module):
        def __init__(self, state_size, action_size, hidden_size=128):
            super(DQN, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, action_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)
    return DQN,


@app.cell
def _():
    # DQN Agent
    class DQNAgent:
        def __init__(
            self,
            state_size,
            action_size,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=64,
            hidden_size=128,
        ):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=memory_size)
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.gamma = gamma
            self.batch_size = batch_size

            # Device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Neural networks
            self.q_network = DQN(state_size, action_size, hidden_size).to(self.device)
            self.target_network = DQN(state_size, action_size, hidden_size).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

            # Initialize target network
            self.update_target_network()

        def update_target_network(self):
            self.target_network.load_state_dict(self.q_network.state_dict())

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def act(self, state, training=True):
            if training and np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().data.numpy().argmax()

        def replay(self):
            if len(self.memory) < self.batch_size:
                return

            batch = random.sample(self.memory, self.batch_size)
            states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
            actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
            next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
            dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)

            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        def save(self, filepath):
            torch.save(self.q_network.state_dict(), filepath)

        def load(self, filepath):
            self.q_network.load_state_dict(torch.load(filepath))
            self.update_target_network()
    return DQNAgent,


@app.cell
def _():
    # Training function
    def train_agent(
        agent: DQNAgent,
        env: gym.Env,
        episodes: int,
        max_steps: int = 500,
        update_target_every: int = 10,
    ) -> Tuple[List[float], List[int], List[float]]:
        episode_rewards = []
        episode_lengths = []
        episode_losses = []
        recent_rewards = deque(maxlen=100)

        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            total_loss = 0
            steps = 0

            for step in range(max_steps):
                action = agent.act(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Modify reward to encourage longer episodes
                if done and step < 499:
                    reward = -10

                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

                if len(agent.memory) > agent.batch_size:
                    agent.replay()

                if done:
                    break

            # Update target network periodically
            if episode % update_target_every == 0:
                agent.update_target_network()

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            recent_rewards.append(total_reward)

            # Calculate average loss (simplified)
            if len(agent.memory) > agent.batch_size:
                episode_losses.append(agent.epsilon)  # Use epsilon as proxy for learning progress

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(recent_rewards)
                print(
                    f"Episode {episode + 1}/{episodes} - "
                    f"Avg Reward: {avg_reward:.2f} - "
                    f"Epsilon: {agent.epsilon:.3f} - "
                    f"Steps: {steps}"
                )

        return episode_rewards, episode_lengths, episode_losses
    return train_agent,


@app.cell
def _():
    # Create environment and agent
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        hidden_size=128,
    )
    return action_size, agent, env, state_size


@app.cell
def _():
    # Training controls
    _training_episodes = mo.ui.slider(
        1, 500, value=200, label="Training Episodes"
    )
    _start_training = mo.ui.button(
        value=False, label="Start Training", on_click=lambda _: True
    )
    _reset_agent = mo.ui.button(
        value=False, label="Reset Agent", on_click=lambda _: True
    )
    return _reset_agent, _start_training, _training_episodes


@app.cell
def _(_reset_agent, _start_training, _training_episodes, agent, env, state_size, action_size):
    # Training state
    _training_complete = False
    _episode_rewards = []
    _episode_lengths = []
    _episode_losses = []

    if _start_training.value:
        _start_training.value = False  # Reset button
        _training_complete = False
        
        # Reset agent if requested
        if _reset_agent.value:
            agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=0.001,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                memory_size=10000,
                batch_size=64,
                hidden_size=128,
            )
            _reset_agent.value = False

        # Train agent
        _episode_rewards, _episode_lengths, _episode_losses = train_agent(
            agent, env, episodes=_training_episodes.value
        )
        _training_complete = True

    return (
        _episode_lengths,
        _episode_losses,
        _episode_rewards,
        _training_complete,
    )


@app.cell
def _(_episode_rewards, _training_complete, mo):
    # Dashboard: Learning Progress
    if _training_complete and len(_episode_rewards) > 0:
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Episode Rewards",
                "Moving Average (100 episodes)",
                "Episode Lengths",
                "Learning Progress",
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
        )

        episodes = list(range(1, len(_episode_rewards) + 1))

        # Episode Rewards
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=_episode_rewards,
                mode="lines",
                name="Reward",
                line=dict(color="blue", width=1),
            ),
            row=1,
            col=1,
        )

        # Moving Average
        window = 100
        if len(_episode_rewards) >= window:
            moving_avg = [
                np.mean(_episode_rewards[max(0, i - window + 1) : i + 1])
                for i in range(len(_episode_rewards))
            ]
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=moving_avg,
                    mode="lines",
                    name=f"{window}-episode Average",
                    line=dict(color="red", width=2),
                ),
                row=1,
                col=2,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=_episode_rewards,
                    mode="lines",
                    name="Reward",
                    line=dict(color="blue", width=1),
                ),
                row=1,
                col=2,
            )

        # Episode Lengths
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=_episode_lengths,
                mode="lines",
                name="Length",
                line=dict(color="green", width=1),
            ),
            row=2,
            col=1,
        )

        # Learning Progress (Success Rate)
        success_threshold = 195  # CartPole-v1 solved threshold
        window_size = 20
        success_rates = []
        for i in range(len(_episode_rewards)):
            start_idx = max(0, i - window_size + 1)
            recent = _episode_rewards[start_idx : i + 1]
            success_rate = sum(1 for r in recent if r >= success_threshold) / len(recent) * 100
            success_rates.append(success_rate)

        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=success_rates,
                mode="lines",
                name="Success Rate (%)",
                line=dict(color="purple", width=2),
                fill="tozeroy",
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="CartPole DQN Learning Dashboard",
            showlegend=True,
        )

        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_xaxes(title_text="Episode", row=2, col=2)
        fig.update_yaxes(title_text="Reward", row=1, col=1)
        fig.update_yaxes(title_text="Average Reward", row=1, col=2)
        fig.update_yaxes(title_text="Steps", row=2, col=1)
        fig.update_yaxes(title_text="Success Rate (%)", row=2, col=2)

        _dashboard = mo.ui.plotly(fig)
    else:
        _dashboard = mo.md("**Training dashboard will appear here after training starts.**")

    return _dashboard,


@app.cell
def _(_episode_rewards, _training_complete):
    # Statistics
    if _training_complete and len(_episode_rewards) > 0:
        _stats = {
            "Total Episodes": len(_episode_rewards),
            "Average Reward": f"{np.mean(_episode_rewards):.2f}",
            "Max Reward": f"{np.max(_episode_rewards):.2f}",
            "Min Reward": f"{np.min(_episode_rewards):.2f}",
            "Final 100 Avg": f"{np.mean(_episode_rewards[-100:]):.2f}" if len(_episode_rewards) >= 100 else "N/A",
            "Episodes to Solve (>195)": next(
                (i + 1 for i, r in enumerate(_episode_rewards) if r >= 195), "Not solved"
            ),
        }
    else:
        _stats = {"Status": "No training data yet"}
    return _stats,


@app.cell
def _(_stats, mo):
    # Display statistics
    _stats_display = mo.md(
        "\n".join([f"**{k}:** {v}" for k, v in _stats.items()])
    )
    return _stats_display,


@app.cell
def _():
    # Test agent button
    _test_agent = mo.ui.button(
        value=False, label="Test Trained Agent", on_click=lambda _: True
    )
    return _test_agent,


@app.cell
def _(_test_agent, _training_complete, agent, env, go, gym, mo, np):
    # Test the trained agent
    _test_results = None
    if _test_agent.value and _training_complete:
        _test_agent.value = False
        
        # Run a test episode
        test_env = gym.make("CartPole-v1", render_mode=None)
        state, _ = test_env.reset()
        total_reward = 0
        steps = 0
        states_history = [state.copy()]
        
        for step in range(500):
            action = agent.act(state, training=False)  # No exploration during test
            state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            states_history.append(state.copy())
            total_reward += reward
            steps += 1
            if done:
                break
        
        test_env.close()
        
        _test_results = {
            "Test Reward": total_reward,
            "Test Steps": steps,
            "Status": "✅ Solved!" if total_reward >= 195 else "⚠️ Not solved yet",
        }
        
        # Create visualization of state trajectory
        states_array = np.array(states_history)
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=states_array[:, 0],
            mode='lines',
            name='Cart Position',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            y=states_array[:, 1],
            mode='lines',
            name='Cart Velocity',
            line=dict(color='green', width=2)
        ))
        fig.add_trace(go.Scatter(
            y=states_array[:, 2],
            mode='lines',
            name='Pole Angle',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            y=states_array[:, 3],
            mode='lines',
            name='Pole Angular Velocity',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title="Test Episode: State Trajectory",
            xaxis_title="Step",
            yaxis_title="State Value",
            height=400,
            showlegend=True,
        )
        
        _test_visualization = mo.ui.plotly(fig)
    elif _test_agent.value and not _training_complete:
        _test_agent.value = False
        _test_results = {"Status": "Please train the agent first!"}
        _test_visualization = mo.md("Train the agent before testing.")
    else:
        _test_results = None
        _test_visualization = mo.md("Click 'Test Trained Agent' to see it in action!")
    
    return _test_results, _test_visualization,


@app.cell
def _(_dashboard, _stats_display, _start_training, _training_episodes, _reset_agent, _test_agent, _test_results, _test_visualization, mo):
    # Main UI Layout
    mo.vstack(
        [
            mo.md("# 🎯 CartPole Reinforcement Learning Dashboard"),
            mo.md(
                "This notebook demonstrates Deep Q-Network (DQN) learning on the classic CartPole problem. "
                "Watch how quickly the neural network learns to balance the pole!"
            ),
            mo.hstack(
                [
                    _training_episodes,
                    _start_training,
                    _reset_agent,
                    _test_agent,
                ],
                justify="start",
                gap="1rem",
            ),
            mo.md("## 📊 Training Statistics"),
            _stats_display,
            mo.md("## 📈 Learning Progress Dashboard"),
            _dashboard,
            mo.md("## 🎮 Test Trained Agent"),
            mo.md("\n".join([f"**{k}:** {v}" for k, v in (_test_results or {}).items()])),
            _test_visualization,
        ],
        gap="1rem",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
