#neuro_prime.py


# Import the necessary modules
from math_problem_verifier import MathProblemVerifier
from star_model import STaRModel
from process_supervised_reward_model import ProcessSupervisedRewardModel
from q_learning_agent import QLearningAgent

def integrated_system(problem):
    # Step 1: Generate and Verify Solutions
    verifier = MathProblemVerifier(model_name='gpt-3')
    solutions = verifier.generate_solutions(problem)
    best_solution = verifier.select_best_solution(problem, solutions)

    # Step 2: Generate Rationale
    star_model = STaRModel(model_name='gpt-4', tokenizer_name='gpt-4-tokenizer')
    rationale = star_model.rationalize(problem, best_solution)

    # Step 3: Step-by-Step Evaluation
    prm = ProcessSupervisedRewardModel(model_name='gpt-4')
    score = prm.score_solution(best_solution)
    
    
    # Step 4: Step-by-Step Evaluation
    q_agent = QLearningAgent(state_size=768, action_size=10, learning_rate=0.001, discount_factor=0.95, epsilon_decay=0.995)

    return {
        'problem': problem,
        'solution': best_solution,
        'rationale': rationale,
        'step_by_step_score': score
    }



# Example usage
if __name__ == "__main__":
    problem = "What is 2 + 2?"
    result = integrated_system(problem)
    print(result)
