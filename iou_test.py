def compute_iou(interval_a, interval_b):
    """
    计算两个时间窗口的IoU
    """
    start_a, end_a = interval_a
    start_b, end_b = interval_b

    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = (end_a - start_a) + (end_b - start_b) - intersection
    print(intersection, union)
    return intersection / union if union != 0 else 0

def compute_average_iou(list_a, list_b):
    """
    计算两个列表的平均IoU
    """
    total_iou = 0
    count = 0
    
    # 计算每个预测时间窗口和真实时间窗口之间的IoU
    for interval_a in list_a:
        max_iou = 0
        for interval_b in list_b:
            iou = compute_iou(interval_a, interval_b)
            if iou > max_iou:
                max_iou = iou
        total_iou += max_iou
        count += 1
    
    # 如果没有任何交集，IoU默认为0
    if count == 0:
        return 0
    
    return total_iou / count

def compute_iou_loss(predicted_list, true_list):
    """
    计算IoU Loss，IoU Loss = 1 - 平均IoU
    """
    average_iou = compute_average_iou(predicted_list, true_list)
    print(average_iou)
    return 1 - average_iou

# 示例列表
predicted_list = [[2, 3], [4, 5]]
true_list = [[1, 2], [4, 6], [7, 8]]

# 计算IoU Loss
iou_loss = compute_iou_loss(predicted_list, true_list)
print("IoU Loss:", iou_loss)
