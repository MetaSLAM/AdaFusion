import numpy as np
from sklearn.neighbors import KDTree


def recall_atN(features: np.ndarray, pos_items: list, N: int, Lp: int):
    """compute the Recall-@N for the network performance.
    Args:
        features: Features of all test items. np.ndarray of size [N,D]
        pos_items: Relation of the positive items. See file `robotcar_makePair.py`
        N: param of Recall-@N
        Lp: distance metric, norm Lp
    Returns:
        recall: The average Recall-@N value
    """
    success_num = 0
    total_num = len(pos_items)
    tree = KDTree(features, p=Lp)
    ind_n1 = tree.query(features, k=N + 1, return_distance=False)

    for i in range(len(ind_n1)):
        if not pos_items[i]:  # skip empty
            total_num -= 1
            continue

        retrieved_items = set(np.setdiff1d(ind_n1[i], [i]))  # array([, , ,]), size [N]
        if retrieved_items & set(pos_items[i]):
            success_num += 1

    return success_num / total_num


def recall_atN_with_error_record(
    features: np.ndarray, pos_items: list, N: int, Lp: int
):
    """compute the Recall-@N for the network performance. This function will also record
    the incorrectly retrieved items so that it is useful for debugging.
    Args:
        features: Features of all test items. np.ndarray of size [N,D]
        pos_items: Relation of the positive items. See file `robotcar_makePair.py`
        N: param of Recall-@N
        Lp: distance metric, norm Lp
    Returns:
        recall: The average Recall-@N value
        err_record: The incorrectly retrieved items, [(item_id, retrieved_ids, gt_id),(),...]
    """
    success_num = 0
    err_record = []
    total_num = len(pos_items)
    tree = KDTree(features, p=Lp)
    ind_n1 = tree.query(features, k=N + 1, return_distance=False)

    for i in range(len(ind_n1)):
        if not pos_items[i]:  # skip empty
            total_num -= 1
            continue

        retrieved_items = set(np.setdiff1d(ind_n1[i], [i]))  # array([, , ,]), size [N]
        if retrieved_items & set(pos_items[i]):
            success_num += 1
        else:
            err_tp = (i, tuple(retrieved_items), tuple(pos_items[i]))
            err_record.append(err_tp)

    return success_num / total_num, err_record

