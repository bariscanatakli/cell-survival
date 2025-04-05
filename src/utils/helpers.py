def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def one_hot_encode(index, size):
    encoding = [0] * size
    encoding[index] = 1
    return encoding

def log_metrics(metrics):
    import pandas as pd
    df = pd.DataFrame(metrics)
    df.to_csv('metrics_log.csv', index=False)