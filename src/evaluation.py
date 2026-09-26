def calculate_macro_f05(y_true: dict, y_pred: dict) -> float:
    """
    Calculates macro-averaged F0.5 score.
    y_true: dict of entity_id -> set of matched entity_ids (excluding self)
    y_pred: dict of entity_id -> set of matched entity_ids (excluding self)
    """
    f05_scores = []
    
    for entity_id in y_true.keys():
        true_matches = y_true.get(entity_id, set())
        pred_matches = y_pred.get(entity_id, set())
        
        if len(true_matches) == 0 and len(pred_matches) == 0:
            f05_scores.append(1.0)
            continue
        elif len(true_matches) == 0 and len(pred_matches) > 0:
            f05_scores.append(0.0)
            continue
            
        tp = len(true_matches.intersection(pred_matches))
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision == 0 and recall == 0:
            f05 = 0.0
        else:
            # F0.5 = (1 + 0.5^2) * (precision * recall) / ((0.5^2 * precision) + recall)
            f05 = 1.25 * (precision * recall) / (0.25 * precision + recall)
            
        f05_scores.append(f05)
        
    return sum(f05_scores) / len(f05_scores) if f05_scores else 0.0
