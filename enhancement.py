import torch
import torch.nn as nn


class AdaptiveEnhancement(nn.Module):
    """
    Adaptive enhancement module.

    Uses:
        - original RGB image
        - illumination map
        - CNN-predicted enhancement parameters

    Dark regions receive stronger enhancement.
    Bright regions receive weaker enhancement.
    """

    def __init__(self):
        super().__init__()

    def forward(self, rgb, illumination, parameters):
        """
        Parameters
        ----------
        rgb : [B, 3, H, W]
            Original RGB image.

        illumination : [B, 1, H, W]
            Estimated illumination map.

        parameters : [B, 3, H, W]
            Enhancement parameter maps predicted by CNN.

        Returns
        -------
        enhanced : [B, 3, H, W]
            Enhanced image.
        """

        # ---------------------------------------------------------
        # 1. Normalize illumination to [0, 1]
        # ---------------------------------------------------------
        illumination = torch.clamp(illumination, 0.0, 1.0)

        # ---------------------------------------------------------
        # 2. Dark-region weight
        #
        # Low illumination -> high enhancement weight
        # High illumination -> low enhancement weight
        # ---------------------------------------------------------
        dark_weight = 1.0 - illumination

        # ---------------------------------------------------------
        # 3. Convert CNN output into positive enhancement strength
        # ---------------------------------------------------------
        strength = torch.sigmoid(parameters)

        # ---------------------------------------------------------
        # 4. Combine predicted strength with illumination
        # ---------------------------------------------------------
        adaptive_gain = strength * dark_weight

        # ---------------------------------------------------------
        # 5. Brightness enhancement
        #
        # I_enh = I + gain * (1-I)
        #
        # This increases intensity while keeping values bounded.
        # ---------------------------------------------------------
        enhanced = rgb + adaptive_gain * (1.0 - rgb)

        # ---------------------------------------------------------
        # 6. Keep valid image range
        # ---------------------------------------------------------
        enhanced = torch.clamp(enhanced, 0.0, 1.0)

        return enhanced


if __name__ == "__main__":

    print("=" * 60)
    print("Adaptive Enhancement Module Test")
    print("=" * 60)

    # Dummy input
    rgb = torch.rand(1, 3, 256, 256)

    illumination = torch.rand(1, 1, 256, 256)

    parameters = torch.randn(1, 3, 256, 256)

    enhancer = AdaptiveEnhancement()

    enhanced = enhancer(
        rgb,
        illumination,
        parameters
    )

    print("RGB shape          :", rgb.shape)
    print("Illumination shape :", illumination.shape)
    print("Parameter shape    :", parameters.shape)
    print("Enhanced shape     :", enhanced.shape)

    print("Enhanced min       :", enhanced.min().item())
    print("Enhanced max       :", enhanced.max().item())

    print("=" * 60)