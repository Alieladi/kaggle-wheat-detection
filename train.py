import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T

def get_model(num_classes):
    """args:
        num_classes: number of classes e.g. 2 for 1 class (wheat head) + background
    """
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main(root, model, optimizer, num_epoch, last_epoch):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # use our dataset and defined transformations
    dataset = WheatDataset(root, get_transform(train=True))
    dataset_test = WheatDataset(root, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # move model to the right device
    model.to(device)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for epochs
    for epoch in range(last_epoch, last_epoch + num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device)

    return model, optimizer, last_epoch + num_epochs
    print("That's it!")


if __name__="__main__":
    # initialize model and optimizer
    num_classes = 2 # 1 class (wheat head) + background
    model = get_model(num_classes)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # load existing checkpoint (comment if none existing)
    from_epoch = 10
    PATH = f"model_weights_v1_/model_weights_v1_{from_epoch}.tar"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # train
    root = "global-wheat-detection/"
    num_epochs = 1
    model, optimizer, last_epoch = main(root, model, optimizer, num_epochs, from_epoch)

    # save model and optimizer weights
    PATH = f"model_weights_v1/model_weights_v1_{last_epoch}.tar"
    #torch.save(model.state_dict(), PATH with .pt)
    torch.save({
                'epoch': last_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, PATH)
