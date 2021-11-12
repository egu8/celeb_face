import transforms as T
from fine_tune import get_model_object_segmentation
from kaggle_dataset import KaggleDataset
import torch

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

from engine import train_one_epoch, evaluate
import utils


def train():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 5
    # use our dataset and defined transformations
    dataset = KaggleDataset('data/train', get_transform(train=True))
    dataset_test = KaggleDataset('data/val', get_transform(train=False))

    # dataset = torch.utils.data.Subset(dataset, torch.arange(5))
    # dataset_test = torch.utils.data.Subset(dataset_test, torch.arange(2))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_object_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        # Save model weights
        torch.save(model.state_dict(), "saved_weights/RNN_detector_epoch_{}".format(epoch))


MODEL_WEIGHTS = "saved_weights/RNN_detector_epoch_6"

def evaluate_pic(pic):
    # get the model using our helper function
    model = get_model_object_segmentation(5)

    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model.eval()

    x = T.ToTensor()(pic)[0].unsqueeze_(0)
    predictions = model(x)

    print(predictions)


if __name__ == "__main__":
    from PIL import Image
    img = Image.open("data/train/pictures/httpgonetworthcomwpcontentuploadsthumbsjpg.jpg").convert("RGB")
    evaluate_pic(img)

    # train()
    # model = get_model_object_segmentation(5)

    # dataset = KaggleDataset('data/train', get_transform(train=True))
    # data_loader = torch.utils.data.DataLoader(
    # dataset, batch_size=2, shuffle=True,
    # collate_fn=utils.collate_fn)

    # # For Training
    # images,targets = next(iter(data_loader))

    # images = list(image for image in images)
    # targets = [{k: v for k, v in t.items()} for t in targets]
    # output = model(images,targets)   # Returns losses and detections

    # print(output)

    # # For inference
    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # predictions = model(x)           # Returns predictions

    # print(predictions)