import torchvision


def build_model(name, num_classes=10, pretrained=False):
    """Model factory. 'vgg16' -> VGG16-BN (CIFAR-10, per paper/EDL literature);
    'resnet18'/'resnet50' -> torchvision (CIFAR-100 / fine-grained)."""
    if name == "vgg16":
        return torchvision.models.vgg16_bn(weights=None, num_classes=num_classes)
    if name == "resnet18":
        return torchvision.models.resnet18(weights=None, num_classes=num_classes)
    if name == "resnet50":
        return torchvision.models.resnet50(weights=None, num_classes=num_classes)
    raise ValueError(name)
