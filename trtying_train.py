import torch
import utils
from model import model_init
from dataset import ChallengeDataset
model = model_init()
dataset = ChallengeDataset('dl_challenge/')
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn
)

# For Training
images, targets = next(iter(data_loader))
images = list(image.to(device='cuda') for image in images)
targets = [{k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]


# write the same line of targets, but in cuda

# model = model.float()
model.to(device='cuda')
output = model(images, targets)  # Returns losses and detections

# output to cpu

raise Exception('Stop here')
# For inference
model.eval()
x = [torch.rand(6, 300, 400), torch.rand(6, 500, 400)]
print(x[0].dtype)
predictions = model(x)  # Returns predictions
print(predictions[0])