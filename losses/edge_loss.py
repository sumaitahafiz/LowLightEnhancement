import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def get_edges(self, image):

        gray = (
            0.299 * image[:, 0:1, :, :] +
            0.587 * image[:, 1:2, :, :] +
            0.114 * image[:, 2:3, :, :]
        )

        grad_x = F.conv2d(
            gray,
            self.sobel_x,
            padding=1
        )

        grad_y = F.conv2d(
            gray,
            self.sobel_y,
            padding=1
        )

        edge = torch.sqrt(
            grad_x ** 2 +
            grad_y ** 2 +
            1e-6
        )

        return edge

    def forward(self, original, enhanced):

        original_edges = self.get_edges(original)
        enhanced_edges = self.get_edges(enhanced)

        return torch.mean(
            torch.abs(enhanced_edges - original_edges)
        )


if __name__ == "__main__":

    print("=" * 60)
    print("EDGE LOSS TEST")
    print("=" * 60)

    original = torch.rand(1, 3, 256, 256)
    enhanced = torch.rand(1, 3, 256, 256)

    loss_function = EdgeLoss()

    loss = loss_function(
        original,
        enhanced
    )

    print("Original shape :", original.shape)
    print("Enhanced shape :", enhanced.shape)
    print("Loss value     :", loss.item())

    print("=" * 60)
    print("EDGE LOSS TEST COMPLETED")
    print("=" * 60)