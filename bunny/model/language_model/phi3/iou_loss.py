def interval_intersection(interval_a, interval_b):
    """
    计算两个时间窗口的交集长度
    """
    start_a, end_a = interval_a
    start_b, end_b = interval_b

    if end_a < start_b or end_b < start_a:
        # No overlap
        return 0
    return min(end_a, end_b) - max(start_a, start_b)

def compute_total_intersection(list_a, list_b):
    """
    计算两个列表中所有时间窗口的交集总长度
    """
    total_intersection = 0
    
    for interval_a in list_a:
        for interval_b in list_b:
            total_intersection += interval_intersection(interval_a, interval_b)
    
    return total_intersection

def compute_total_union(list_a, list_b):
    """
    计算两个列表中所有时间窗口的并集总长度
    """
    # 首先计算单个列表的总长度
    total_length_a = sum(end - start for start, end in list_a)
    total_length_b = sum(end - start for start, end in list_b)

    # 然后减去交集的长度
    total_intersection = compute_total_intersection(list_a, list_b)
    
    total_union = total_length_a + total_length_b - total_intersection
    
    return total_union

def compute_iou(list_a, list_b):
    """
    计算两个列表的IoU
    """
    total_intersection = compute_total_intersection(list_a, list_b)
    total_union = compute_total_union(list_a, list_b)
    print(total_intersection, total_union)
    
    if total_union == 0:
        return 0
    
    return 1 - (total_intersection / total_union)

if __name__ == "__main__":
    predicted_list = [[2, 3], [4, 5]]
    true_list = [[1, 2], [4, 6], [7, 8]]

    iou = compute_iou(predicted_list, true_list)
    print("IoU:", iou)
