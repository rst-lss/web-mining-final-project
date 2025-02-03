import random
from collections import deque
import numpy as np

def calculate_discounted_reward(position, relevance_score):
    """
    Calculate the discounted reward for a document at a given position.
    
    Args:
        position (int): Position in the ranking sequence (0-based)
        relevance_score (float): Relevance score of the document
        
    Returns:
        float: Discounted reward value. Returns 0 for position 0,
              otherwise returns relevance_score / log2(position + 1)
    """
    if position == 0:
        return 0
    return relevance_score / np.log2(position + 1)


class SearchState:
    """
    Represents the state of a search ranking sequence.
    
    Attributes:
        position (int): Current position in the ranking sequence
        query_id: Identifier for the current query
        available_docs (list): List of remaining documents to be ranked
    """
    def __init__(self, position, query_id, available_docs):
        self.position = position
        self.query_id = query_id
        self.available_docs = available_docs

    def get_next_doc(self):
        """Remove and return the next available document."""
        return self.available_docs.pop()

    def is_initial(self):
        """Check if this is the initial state (position 0)."""
        return self.position == 0

    def is_terminal(self):
        """Check if this is a terminal state (no more documents available)."""
        return len(self.available_docs) == 0


class ExperienceBuffer:
    """
    A circular buffer for storing and sampling experience tuples for reinforcement learning.
    
    Each experience consists of (state, action, reward, next_state, done).
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.experiences = deque(maxlen=capacity)

    def add_experience(self, state, action, reward, next_state, done):
        """Add a single experience tuple to the buffer."""
        experience = (state, action, np.array([reward]), next_state, done)
        self.experiences.append(experience)

    def add_experiences_from_dataframe(self, dataframe, num_sequences):
        """
        Generate and add multiple experience sequences from a dataframe.
        
        Args:
            dataframe: DataFrame containing query and document information
            num_sequences (int): Number of sequences to generate
        """
        for _ in range(num_sequences):
            # Select random query
            query_id = random.choice(dataframe["qid"].unique())
            query_df = dataframe[dataframe["qid"] == int(query_id)].reset_index()
            
            # Generate random document ordering
            document_ids = [row["doc_id"] for _, row in query_df.iterrows()]
            sequence_order = list(range(len(query_df)))
            random.shuffle(sequence_order)
            
            # Create experiences for the sequence
            for position, idx in enumerate(sequence_order):
                current_row = query_df.iloc[idx]
                
                current_state = SearchState(position, current_row["qid"], document_ids[:])
                document_id = current_row["doc_id"]
                next_state = SearchState(position + 1, current_row["qid"], document_ids[:])
                reward = calculate_discounted_reward(position + 1, current_row["rank"])
                is_terminal = position + 1 == len(sequence_order)
                
                self.add_experience(current_state, document_id, reward, next_state, is_terminal)

    def sample_batch(self, batch_size):
        """
        Sample a random batch of experiences from the buffer.
        
        Returns:
            tuple: (states, actions, rewards, next_states, done_flags)
        """
        batch = random.sample(self.experiences, batch_size)
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.experiences)