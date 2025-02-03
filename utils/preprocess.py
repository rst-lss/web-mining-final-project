import numpy as np
import pandas as pd

def load_letor(file_path):
    """Load the LETOR dataset into a DataFrame.

    Parameters:
    file_path (str): Path to the dataset file.

    Returns:
    pd.DataFrame: Processed dataset.
    """
    data = {i: [] for i in range(1, 47)}
    data.update({"qid": [], "rank": [], "doc_id": []})

    with open(file_path, "r") as file:
        for line_count, line in enumerate(file, start=1):
            if line_count % 60000 == 0:
                print(f"Processed {line_count} lines")

            features, doc_info = line.strip().split("#")
            doc_id = doc_info.split("=")[1][:-5]
            data["doc_id"].append(doc_id)

            for index, item in enumerate(features.split()):
                if index == 0:
                    data["rank"].append(int(item))  # Rank (first value)
                elif index == 1:
                    data["qid"].append(int(item.split(":")[1]))  # Query ID
                elif ":" in item:
                    feature_index, value = item.split(":")
                    data[int(feature_index)].append(float(value))

    df = pd.DataFrame(data).sort_values(["qid", "rank"], ascending=False)
    return df

def get_model_inputs(state, action, dataset):
    """Generate input features for a model.

    Parameters:
    state (object): Current state.
    action (str): Selected action.
    dataset (pd.DataFrame): Dataset containing features.

    Returns:
    np.ndarray: Feature vector.
    """
    return np.array([state.t] + extract_features(state.qid, action, dataset))

def get_multiple_model_inputs(state, doc_list, dataset):
    """Generate input features for multiple documents.

    Parameters:
    state (object): Current state.
    doc_list (list): List of document IDs.
    dataset (pd.DataFrame): Dataset containing features.

    Returns:
    np.ndarray: Feature matrix.
    """
    return np.insert(
        extract_query_features(state.qid, doc_list, dataset), 0, state.t, axis=1
    )

def extract_features(qid, doc_id, dataset):
    """Extract feature values for a specific query-document pair.

    Parameters:
    qid (int): Query ID.
    doc_id (str): Document ID.
    dataset (pd.DataFrame): Processed dataset.

    Returns:
    list: Feature values.
    """
    qid, doc_id = int(qid), str(doc_id)
    filtered_data = dataset[(dataset["doc_id"].str.contains(doc_id)) & (dataset["qid"] == qid)]
    
    if filtered_data.empty:
        raise ValueError("Invalid dataset: No matching records found.")

    return filtered_data.drop(columns=["qid", "doc_id", "rank"]).values.tolist()[0]

def extract_query_features(qid, doc_list, dataset):
    """Extract features for multiple documents under a query.

    Parameters:
    qid (int): Query ID.
    doc_list (list): List of document IDs.
    dataset (pd.DataFrame): Processed dataset.

    Returns:
    np.ndarray: Feature matrix.
    """
    qid = int(qid)
    doc_set = set(doc_list)

    filtered_data = dataset[dataset["qid"] == qid]
    if doc_set:
        filtered_data = filtered_data[filtered_data["doc_id"].isin(doc_set)]
    
    if filtered_data.empty:
        raise ValueError("Invalid dataset: No matching records found.")

    return filtered_data.drop(columns=["qid", "doc_id", "rank"]).values
