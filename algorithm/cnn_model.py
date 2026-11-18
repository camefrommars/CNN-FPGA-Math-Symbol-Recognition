
import torch
import torch.nn as nn


class MathSymbolCNN(nn.Module):

    def __init__(self):
        super(MathSymbolCNN, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=0
        )

        # Second convolution layer
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=0
        )

        # Activation and pooling
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Fully connected layer
        self.fc = nn.Linear(
            in_features=288,
            out_features=5
        )

    def forward(self, x):

        # Convolution 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Convolution 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Classification
        x = self.fc(x)

        return x


if __name__ == "__main__":

    model = MathSymbolCNN()

    # Test using one 32x32 grayscale image
    dummy_input = torch.randn(1, 1, 32, 32)

    output = model(dummy_input)

    print("CNN Model Architecture:")
    print(model)

    print("\nInput shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    print("\nCNN model created successfully!")
