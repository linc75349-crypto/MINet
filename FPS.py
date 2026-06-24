import numpy as np
import torch
import time
from tqdm import tqdm

from config import transform as transform_cfg
from model import MINet


@torch.no_grad()
def compute_speed(model, device="cuda"):
    """Measure inference speed with random inputs."""
    batch_size = 300
    inputs = torch.randn(batch_size, 3, transform_cfg["rescale_size"], transform_cfg["rescale_size"])

    if device == "cuda":
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    time_spent = []
    for idx in tqdm(range(100)):
        start_time = time.time()
        _ = model(inputs)

        if device == "cuda":
            torch.cuda.synchronize()

        if idx > 20:  # skip warm-up
            time_spent.append(time.time() - start_time)

    avg_fps = batch_size / np.mean(time_spent)
    print(f"Average speed: {avg_fps:.4f} fps")
    return avg_fps


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    model = MINet()
    compute_speed(model)
