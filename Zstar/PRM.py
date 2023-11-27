import torch
from transformers import AutoModel, AutoTokenizer

class ProcessSupervisedRewardModel:
    def __init__(self, model_name):
        """
        Initialize the Process Supervised Reward Model.
        
        :param model_name: Name of the pre-trained language model (e.g., 'gpt-4')
        """
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_data(self, labeled_data):
        """
        Preprocess the labeled data for training.
        
        :param labeled_data: List of tuples (solution, step_labels) where
                             solution is a string of the step-by-step solution,
                             and step_labels are labels for each step (1 for correct, 0 for incorrect)
        :return: Processed data suitable for model training
        """
        processed_data = []
        for solution, step_labels in labeled_data:
            tokens = solution.split('\n')
            for step, label in zip(tokens, step_labels):
                input_ids = self.tokenizer.encode(step, return_tensors='pt')
                processed_data.append((input_ids, torch.tensor([label], dtype=torch.float)))
        return processed_data

    def train(self, training_data, learning_rate=1e-5, epochs=2):
        """
        Train the PRM on the provided labeled data.
        
        :param training_data: List of preprocessed training data
        :param learning_rate: Learning rate for the optimizer
        :param epochs: Number of epochs for training
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.model.train()
            for input_ids, label in training_data:
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]  # Get the logits for the last token

                # Compute loss (binary cross-entropy)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

    def score_solution(self, solution):
        """
        Score a solution based on the correctness of each step.

        :param solution: String of the step-by-step solution
        :return: Score of the solution (probability that every step is correct)
        """
        self.model.eval()
        tokens = solution.split('\n')
        probabilities = []

        for step in tokens:
            input_ids = self.tokenizer.encode(step, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]
                probs = torch.sigmoid(logits)  # Convert logits to probabilities
                probabilities.append(probs.item())

        solution_score = 1
        for prob in probabilities:
            solution_score *= prob

        return solution_score
