from engine import train_one_epoch, evaluate
import torch
from dataset import ChallengeDataset, get_transform
from model import model_init
import utils

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = ChallengeDataset('mini_set/', get_transform(train=False))
# dataset_test = ChallengeDataset('data/PennFudanPed', get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[:-50])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=3,
    # shuffle=True,
    num_workers=1,
    collate_fn=utils.collate_fn
)

# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test,
#     batch_size=1,
#     shuffle=False,
#     num_workers=4,
#     collate_fn=utils.collate_fn
# )

# get the model using our helper function
model = model_init()

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(
#     params,
#     lr=1e-5,
#     momentum=0.9,
#     weight_decay=0.0005
# )
optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 100

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    if epoch % 20 == 0:
        torch.save(model.state_dict(), 'model.pth')
    # evaluate on the test dataset
# save the weaitght of the model
    # evaluate(model, data_loader_test, device=device)
print("That's it!")