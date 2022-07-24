import lightly
import torchvision


# Augmentations typically used to train on cifar-10
def cifar10_train_classifier_transforms(input_size=32):
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(input_size, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])


# No additional augmentations for the test set
def cifar10_test_transforms(input_size=32):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])
