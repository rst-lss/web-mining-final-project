import random
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from model.mdp import SearchState, calculate_discounted_reward

# Set random seed for reproducibility
random.seed(2)

class RankingMetrics:
    """
    A class for computing various ranking evaluation metrics including NDCG and correlation coefficients.
    """
    
    @staticmethod
    def compute_dcg(relevance_scores, k, method=0):
        """
        Compute Discounted Cumulative Gain at position k.
        
        Args:
            relevance_scores (list): List of relevance scores
            k (int): Position to compute DCG at
            method (int): DCG computation method (0 or 1)
            
        Returns:
            float: DCG value
        """
        scores = np.asarray(relevance_scores, dtype=float)[:k]
        if not scores.size:
            return 0.0
            
        if method == 0:
            return scores[0] + np.sum(scores[1:] / np.log2(np.arange(2, scores.size + 1)))
        elif method == 1:
            return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
        else:
            raise ValueError("method must be 0 or 1")

    @staticmethod
    def compute_ndcg(relevance_scores, k, method=0):
        """
        Compute Normalized Discounted Cumulative Gain at position k.
        
        Args:
            relevance_scores (list): List of relevance scores
            k (int): Position to compute NDCG at
            method (int): DCG computation method
            
        Returns:
            float: NDCG value
        """
        ideal_dcg = RankingMetrics.compute_dcg(sorted(relevance_scores, reverse=True), k, method)
        if not ideal_dcg:
            return 0.0
        return RankingMetrics.compute_dcg(relevance_scores, k, method) / ideal_dcg

    @staticmethod
    def compute_ndcg_at_k(relevance_scores, k_list):
        """
        Compute NDCG at multiple positions and mean NDCG.
        
        Args:
            relevance_scores (list): List of relevance scores
            k_list (list): List of positions to compute NDCG at
            
        Returns:
            list: NDCG values at specified positions plus mean NDCG
        """
        ndcg_values = []
        running_sum = 0
        
        for i in range(1, len(relevance_scores)):
            current_ndcg = RankingMetrics.compute_ndcg(relevance_scores, i)
            if i in k_list:
                ndcg_values.append(current_ndcg)
            running_sum += current_ndcg
            
        mean_ndcg = running_sum / len(relevance_scores)
        return ndcg_values + [mean_ndcg]

    @staticmethod
    def compute_correlation_metrics(ranking1, ranking2):
        """
        Compute Kendall's Tau and Spearman correlation between two rankings.
        
        Returns:
            tuple: (Kendall's Tau, Spearman correlation)
        """
        return kendalltau(ranking1, ranking2), spearmanr(ranking1, ranking2)


class RankingEvaluator:
    """
    A class for evaluating ranking agents against datasets.
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.metrics = RankingMetrics()

    def get_document_rank(self, query_id, doc_id):
        """Get the true rank of a document for a given query."""
        df = self.dataset[
            (self.dataset["qid"] == query_id) & 
            (self.dataset["doc_id"] == doc_id)
        ]
        return int(df["rank"].iloc[0])

    def convert_docids_to_ranks(self, doc_ids, query_id):
        """Convert a list of document IDs to their corresponding relevance ranks."""
        return [self.get_document_rank(query_id, doc_id) for doc_id in doc_ids]

    def get_agent_ranking(self, agent, query_id):
        """
        Get a complete ranking of documents from the agent for a given query.
        """
        query_docs = self.dataset[self.dataset["qid"] == int(query_id)].reset_index()
        available_docs = list(query_docs["doc_id"])
        random.shuffle(available_docs)
        
        current_state = SearchState(0, query_id, available_docs)
        ranking = []
        position = 0
        
        while available_docs:
            selected_doc = agent.get_action(current_state, self.dataset)
            position += 1
            available_docs.remove(selected_doc)
            current_state = SearchState(position, query_id, available_docs)
            ranking.append(selected_doc)
            
        return ranking

    def get_true_ranking(self, query_id):
        """Get the true ranking of documents for a query."""
        query_docs = self.dataset[self.dataset["qid"] == query_id]
        return list(query_docs.sort_values("rank", ascending=False)["doc_id"])

    def evaluate_agent(self, agent, ndcg_positions):
        """
        Evaluate an agent's performance across all queries in the dataset.
        
        Args:
            agent: The ranking agent to evaluate
            ndcg_positions (list): Positions at which to compute NDCG
            
        Returns:
            dict: Dictionary containing various evaluation metrics
        """
        unique_queries = set(self.dataset["qid"])
        ndcg_values = np.zeros(len(ndcg_positions) + 1)  # +1 for mean NDCG
        
        for query_id in unique_queries:
            agent_ranking = self.get_agent_ranking(agent, query_id)
            relevance_scores = self.convert_docids_to_ranks(agent_ranking, query_id)
            ndcg_values += np.array(self.metrics.compute_ndcg_at_k(relevance_scores, ndcg_positions))
            
        ndcg_values /= len(unique_queries)
        
        return {
            'ndcg_values': ndcg_values[:-1],
            'mean_ndcg': ndcg_values[-1],
            'positions': ndcg_positions
        }