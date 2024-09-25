from torch import nn
from torchvision.models import resnet50, resnet18, vgg19


class CovidResNet18Model(nn.Module):
    def __init__(self, num_classes=3, train_backbone_params=True):
        super(CovidResNet18Model, self).__init__()
        self.train_backbone_params = train_backbone_params
        self.num_classes = num_classes


        self.backbone = resnet18(pretrained=True)

        # Freeze or unfreeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = self.train_backbone_params

        # Replace the final fully connected layer # Should be 512 for ResNet18
        num_features = self.backbone.fc.in_features  

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes),
        )

        # Assign the modified classifier to the backbone
        self.backbone.fc = self.classifier

    def forward(self, x):
        return self.backbone(x)
    

class CovidResNet50Model(nn.Module):
    def __init__(self, num_classes=3, train_backbone_params=True):
        super(CovidResNet50Model, self).__init__()
        self.train_backbone_params = train_backbone_params
        self.num_classes = num_classes


        self.backbone = resnet50(pretrained=True)

        # Freeze or unfreeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = self.train_backbone_params

        # Replace the final fully connected layer # Should be 512 for ResNet18
        num_features = self.backbone.fc.in_features  

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes),
        )

        # Assign the modified classifier to the backbone
        self.backbone.fc = self.classifier

    def forward(self, x):
        return self.backbone(x)


class CovidVgg19Model(nn.Module):
    def __init__(self, num_classes=3, train_backbone_params=True):
        super(CovidVgg19Model, self).__init__()
        self.train_backbone_params = train_backbone_params
        self.num_classes = num_classes

        self.backbone = vgg19(pretrained=True)

        # Freeze or unfreeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = self.train_backbone_params

        # Correctly access the in_features from the first Linear layer
        num_features = self.backbone.classifier[0].in_features

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes),
        )

        # Assign the modified classifier to the backbone
        self.backbone.classifier = self.classifier

    def forward(self, x):
        return self.backbone(x)
