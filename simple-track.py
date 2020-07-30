import json

import fire
import imageio
import numpy as np
import torch

from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor


def track(
    video_path: str,
    output_path: str = "out.json",
    score_threshold: float = 0.5,
    class_index: int = 1,  # Track people by default
) -> None:
    """Track the objects for a specific class in a given video"""
    # Initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
    reader = imageio.get_reader(video_path)
    # Tracking loop
    active_tracklets = []
    finished_tracklets = []
    for i, frame in enumerate(reader):
        height, width = frame.shape[:2]
        # Detection
        x = to_tensor(frame).to(device)
        result = model(x[None])[0]
        # Feature extraction: (x1, y1, x2, y2) in image coordinates
        # where class == class_index and score > score_threshold
        mask = torch.logical_and(
            result["labels"] == class_index, result["scores"] > score_threshold
        )
        boxes = result["boxes"][mask].data.cpu().numpy() / np.array(
            [width, height, width, height]
        )
        prev_indices = boxes_indices = []
        if i > 0:
            # Pairwise cost: euclidean distance between boxes
            cost = np.linalg.norm(prev_boxes[:, None] - boxes[None], axis=-1)
            # Bipartite matching
            prev_indices, boxes_indices = linear_sum_assignment(cost)
        # Add matches to active tracklets
        for prev_idx, box_idx in zip(prev_indices, boxes_indices):
            active_tracklets[prev_idx]["boxes"].append(
                np.round(boxes[box_idx], 3).tolist()
            )
        # Finalize lost tracklets
        lost_indices = set(range(len(active_tracklets))) - set(prev_indices)
        for lost_idx in sorted(lost_indices, reverse=True):
            finished_tracklets.append(active_tracklets.pop(lost_idx))
        # Activate new tracklets
        new_indices = set(range(len(boxes))) - set(boxes_indices)
        for new_idx in new_indices:
            active_tracklets.append(
                {"start": i, "boxes": [np.round(boxes[new_idx], 3).tolist()]}
            )
        # "Predict" next frame for comparison
        prev_boxes = np.array([tracklet["boxes"][-1] for tracklet in active_tracklets])
    with open(output_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "fps": reader.get_meta_data()["fps"],
                    "tracklets": finished_tracklets + active_tracklets,
                }
            )
        )


if __name__ == "__main__":
    fire.Fire(track)
