"""画像で推定した目標に近いときの停止条件。"""


def should_stabilize(source, position_valid, distance_m, alignment_m):
    return source == 'learned_fine' and position_valid and distance_m <= alignment_m
