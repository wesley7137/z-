class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate, discount_factor, epsilon_decay):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self.create_q_network_with_attention(state_size, action_size)
        self.discount_factor = discount_factor
        self.q_table = torch.zeros(state_size, action_size)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.to(device)
        

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        print("SYSTEM MESSAGE: Epsilon Decay: {}".format(self.epsilon))
        # Ensure epsilon doesn't go below epsilon_min


    def create_q_network_with_attention(self, state_size, action_size):
        # Adjust the input size to match the state size (which is 768 in your case)
        model = nn.Sequential(
            nn.Linear(state_size, 24),  # state_size is now 768
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size)
        )
        print("SYSTEM MESSAGE: Q-Network Initialized")
        return model



    def update(self, state, action, reward, next_state):
        # Convert state and next_state to tensors
        if action < 0 or action >= self.action_size:
            print(f"Invalid action: {action}. Expected range: 0 to {self.action_size - 1}")
            return  # Skip this update
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)

        # Ensure tensors are two-dimensional [batch_size, feature_size]
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if len(next_state_tensor.shape) == 1:
            next_state_tensor = next_state_tensor.unsqueeze(0)
        # Predict Q-values for the current state and the next state
        current_q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)
        # Convert action to tensor and reshape to [batch_size, 1]
        action_tensor = torch.tensor([[action]], dtype=torch.long)
        # Select the Q-value for the action taken
        print("SYSTEM MESSAGE: Q-Learning Current Q: {}".format(current_q_values))
        current_q = current_q_values.gather(1, action_tensor.to(device)).squeeze(-1)
        # Compute the target Q-value
        max_next_q = next_q_values.max(1)[0].detach()
        target_q = reward + self.discount_factor * max_next_q
        print("SYSTEM MESSAGE: Q-Learning Target Q: {}".format(target_q))
        # Compute the loss
        loss = nn.functional.mse_loss(current_q, target_q)
        print("SYSTEM MESSAGE: Q-Learning Loss: {}".format(loss))
        # Backpropagate the loss
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        # Apply updates
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Define a reasonable threshold for Q-value fluctuation
        threshold = 2.0  # Adjust based on Q-value range and training behavior

        # Monitor and log Q-value fluctuations
        if torch.abs(current_q - target_q).mean().item() > threshold:  # Define a reasonable threshold
            print(f"SYSTEM MESSAGE: Large fluctuation in Q-values detected. Current Q: {current_q}, Target Q: {target_q}")

        # Log detailed information for debugging
        print(f"SYSTEM MESSAGE: Action: {action}, Reward: {reward}, Loss: {loss.item()}")
        
        
    def select_action(self, state, epsilon):
        print("SYSTEM MESSAGE: Selecting Action")
        if random.random() < epsilon:
            # Exploration: choose a random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: choose the best action based on Q-values
            with torch.no_grad():
                state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = self.model(state)
                # Ensure q_values is 1D before applying argmax
                action = torch.argmax(q_values.squeeze(), dim=0)[0].item()
                print("SYSTEM MESSAGE: Action Selected: {}".format(action))
                print("Q-values:", q_values)
                print("Selected action:", action)
                return action

                
    def save_model(self, directory, filename):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            filepath = os.path.join(directory, filename)
            torch.save(self.state_dict(), filepath)
            print(f"SYSTEM MESSAGE: Model Saved at {filepath}")
        except Exception as e:
            print(f"SYSTEM MESSAGE: Error saving model: {e}")
    def load_weights(self, filepath):
        self.model.load_state_dict(torch.load(filepath + '_qnetwork.pth'))
        print("SYSTEM MESSAGE: Q-Learning Model Loaded")
