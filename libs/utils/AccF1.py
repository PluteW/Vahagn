


def calculate_accuracy(y_true, y_pred):
    """
    计算准确率
    """
    correct = 0
    total = len(y_true)
    for i in range(total):
        if y_true[i] == y_pred[i]:
            correct += 1
    accuracy = correct / total
    return accuracy

def calculate_precision(y_true, y_pred):
    """
    计算预测率
    """
    true_positive = 0
    false_positive = 0
    total = len(y_true)
    for i in range(total):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                true_positive += 1
            else:
                false_positive += 1
    if true_positive == 0:
        return 0
    precision = true_positive / (true_positive + false_positive)
    return precision

def calculate_recall(y_true, y_pred):
    """
    计算召回率
    """
    true_positive = 0
    false_negative = 0
    total = len(y_true)
    for i in range(total):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                true_positive += 1
            else:
                false_negative += 1
    if true_positive == 0:
        return 0
    recall = true_positive / (true_positive + false_negative)
    return recall

def calculate_f1_score(y_true, y_pred):
    """
    计算F1值
    """
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    if precision == 0 or recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score