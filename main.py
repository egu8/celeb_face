import transforms as T
from fine_tune import get_model_object_segmentation
from kaggle_dataset import KaggleDataset
from celeba_dataset import CelebaDataset
import torch
import matplotlib.pyplot as plt
import torchvision


NUM_CLASSES = 5

MODEL_WEIGHTS = "saved_weights/RNN_detector_best"

identity_mapping = {
    0: "ben_afflek",
    1: "elton_john",
    2: "jerry_seinfeld",
    3: "madonna",
    4: "mindy_kaling"
}

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

    num_classes = NUM_CLASSES
    # # use our dataset and defined transformations
    # dataset = CelebaDataset('img_celeba_data/train', "img_celeba", get_transform(train=True))
    # dataset_test = CelebaDataset('img_celeba_data/val', "img_celeba", get_transform(train=False))

    # use our dataset and defined transformations
    dataset = KaggleDataset('kaggle_data/train', get_transform(train=True))
    dataset_test = KaggleDataset('kaggle_data/val', get_transform(train=False))

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
    # model.load_state_dict(torch.load(MODEL_WEIGHTS))

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)
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

def evaluate_pic(pic):
    # get the model using our helper function
    model = get_model_object_segmentation(NUM_CLASSES)

    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model.eval()

    x = T.ToTensor()(pic)[0].unsqueeze_(0)
    predictions = model(x)

    boxes = predictions[0]['boxes'][0:1]
    labels = predictions[0]['labels'][0:1]
    scores = predictions[0]['scores'][0:1]

    img = x[0] * 255
    img =torch.tensor(img,dtype=torch.uint8)

    str_label = []
    for label in labels:
        str_label.append(identity_mapping[label.item()])

    a=torchvision.utils.draw_bounding_boxes(image=img,boxes=boxes, labels=str_label)
    a=a.permute(1,2,0)
    plt.imshow(a)
    plt.show()


if __name__ == "__main__":

    # Evaluate on one image
    from PIL import Image
    img = Image.open("test_images/jerry-seinfeld-season-10-interview.jpg").convert("RGB")
    evaluate_pic(img)

    # # Train on image set
    # train()