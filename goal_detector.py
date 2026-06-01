import numpy as np
from PIL import Image


class GoalDetector:
    """
    CNN-inspired colour-based goal detector.
    Processes the robot POV camera frame to detect
    whether the green goal zone is visible.

    Pipeline mirrors a CNN:
      Step 1 — Colour channel extraction (like conv filters)
      Step 2 — Spatial downsampling (like max pooling)
      Step 3 — Threshold classification (like fully connected layer)
    """

    # Green goal zone HSV-ish RGB thresholds
    # The goal is a bright semi-transparent green cylinder
    GREEN_R_MAX = 100   # red channel must be low
    GREEN_G_MIN = 150   # green channel must be high
    GREEN_B_MAX = 100   # blue channel must be low

    # Detection parameters
    MIN_GREEN_PIXELS  = 10    # minimum pixels to count as detected
    CONFIDENCE_SCALE  = 5000  # scale factor for confidence score

    def __init__(self):
        self.last_result     = False
        self.last_confidence = 0.0
        self.last_bbox       = None   # (x1, y1, x2, y2) or None

    def detect(self, pil_image):
        """
        Run goal detection on a PIL RGB image.

        Returns:
            detected   (bool)  : True if goal is visible
            confidence (float) : 0.0 to 1.0
            bbox       (tuple) : (x1, y1, x2, y2) or None
        """
        # ── Step 1: Colour channel extraction ────────────────────
        # Convert to numpy — shape (H, W, 3)
        arr = np.array(pil_image, dtype=np.float32)
        R   = arr[:, :, 0]
        G   = arr[:, :, 1]
        B   = arr[:, :, 2]

        # ── Step 2: Spatial max pooling (downsample 4x) ───────────
        # Reduces noise and speeds up processing
        h, w  = R.shape
        R_pool = R[:h//4*4, :w//4*4].reshape(h//4, 4, w//4, 4).max(axis=(1,3))
        G_pool = G[:h//4*4, :w//4*4].reshape(h//4, 4, w//4, 4).max(axis=(1,3))
        B_pool = B[:h//4*4, :w//4*4].reshape(h//4, 4, w//4, 4).max(axis=(1,3))

        # ── Step 3: Green mask — isolate goal colour ──────────────
        green_mask = (
            (R_pool < self.GREEN_R_MAX) &
            (G_pool > self.GREEN_G_MIN) &
            (B_pool < self.GREEN_B_MAX)
        )

        green_pixels = int(np.sum(green_mask))

        # ── Step 4: Bounding box extraction ──────────────────────
        bbox = None
        if green_pixels >= self.MIN_GREEN_PIXELS:
            rows = np.any(green_mask, axis=1)
            cols = np.any(green_mask, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            # Scale back to original image coordinates
            bbox = (x1*4, y1*4, x2*4, y2*4)

        # ── Step 5: Classification ────────────────────────────────
        detected   = green_pixels >= self.MIN_GREEN_PIXELS
        confidence = min(green_pixels / self.CONFIDENCE_SCALE, 1.0)

        self.last_result     = detected
        self.last_confidence = confidence
        self.last_bbox       = bbox

        return detected, confidence, bbox