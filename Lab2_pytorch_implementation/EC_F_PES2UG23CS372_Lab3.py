# lab.py
import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    """
    labels = tensor[:, -1]
    classes, counts = torch.unique(labels, return_counts=True)
    probs = counts.float() / labels.size(0)
    entropy = -torch.sum(probs * torch.log2(probs))
    return entropy.item()


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v))
    """
    labels = tensor[:, -1]
    values, counts = torch.unique(tensor[:, attribute], return_counts=True)
    total = labels.size(0)
    avg_info = 0.0

    for v, count in zip(values, counts):
        subset = tensor[tensor[:, attribute] == v]
        subset_entropy = get_entropy_of_dataset(subset)
        avg_info += (count.item() / total) * subset_entropy

    return avg_info


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)
    """
    total_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    return round(total_entropy - avg_info, 4)


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.
    Returns:
        tuple: ({attr_index: gain, ...}, best_attr_index)
    """
    num_attributes = tensor.size(1) - 1
    gains = {}
    for i in range(num_attributes):
        gains[i] = get_information_gain(tensor, i)
    best_attr = max(gains, key=gains.get)
    return gains, best_attr
