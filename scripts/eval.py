import time
from pathlib import Path

import torch

from models.dqn import DQN, DQNAgent
from utils.eval import evaluate_agent_performance
from utils.preprocess import load_letor

# Configuration constants
DEFAULT_NDCG_POSITIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
FEATURE_DIMENSION = 47
LEARNING_RATE = 3e-4

# Path configurations
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "../../../datasets/MQ2008-list"
MODEL_PATH = BASE_DIR / "../../../build/best-model.pth"


def evaluate_ranking_model(dataset_dir, model_path, folds=range(2, 6)):
    """
    Evaluate a pre-trained ranking model across multiple dataset folds.
    
    Args:
        dataset_dir (Path): Directory containing the dataset folds
        model_path (Path): Path to the pretrained model weights
        folds (iterable): Collection of fold numbers to evaluate
        
    Returns:
        dict: Dictionary mapping fold numbers to their NDCG scores
    """
    start_time = time.time()
    results = {}

    print("Initializing ranking agent with pretrained model...")
    model = DQN(FEATURE_DIMENSION, output_dim=1)
    model.load_state_dict(torch.load(model_path))

    agent = DQNAgent(
        state_dim=FEATURE_DIMENSION,
        learning_rate=LEARNING_RATE,
        buffer=None,
        dataset=None,
        pre_trained_model=model
    )

    for fold in folds:
        print(f"\nEvaluating Fold {fold}")
        
        # Load test dataset for current fold
        test_path = dataset_dir / f"Fold{fold}" / "test.txt"
        test_data = load_letor_dataset(test_path)
        
        # Evaluate model performance
        ndcg_scores = evaluate_agent_performance(
            agent=agent,
            ndcg_positions=DEFAULT_NDCG_POSITIONS,
            test_data=test_data
        )
        
        # Store and display results
        results[fold] = ndcg_scores
        print(f"Fold {fold} NDCG@k values:")
        print(f"Positions: {DEFAULT_NDCG_POSITIONS}")
        print(f"Scores: {ndcg_scores}\n")

    execution_time = time.time() - start_time
    print(f"Model evaluation completed successfully")
    print(f"Total execution time: {execution_time:.2f} seconds")
    
    return results


if __name__ == "__main__":
    evaluate_ranking_model(DATASET_DIR, MODEL_PATH)