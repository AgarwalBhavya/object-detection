import torch

# Dummy conversion (you'd map from VOC annotation to detection format)
def prepare_targets(targets):
    dummy_target = []
    for t in targets:
        dummy_target.append({
            "boxes": torch.tensor([[50, 50, 100, 100]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64)
        })
    return dummy_target
