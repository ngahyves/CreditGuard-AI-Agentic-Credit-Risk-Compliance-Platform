import hashlib

def get_assigned_strategy(client_id: int, weight_b: int = 10) -> str:
    """
    Routes a client to Strategy A or B based on their ID.
    weight_b: percentage of traffic for Strategy B (e.g., 10)
    """
    hash_val = int(hashlib.md5(str(client_id).encode()).hexdigest(), 16)
    return "B" if (hash_val % 100 < weight_b) else "A"