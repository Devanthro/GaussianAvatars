"""
2D animated face renderer for AI face mode.

Loads the static AI face image (glowing curves on black background),
detects the mouth curve, and replaces it each frame with animated
bezier curves driven by blendshape values from audio formant analysis.

Upper lip: smile curve that flattens when jaw opens.
Lower lip: fades in and drops when speaking, creating the mouth opening.
"""

import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = hasattr(cv2, 'polylines') and hasattr(cv2, 'GaussianBlur')
except ImportError:
    _HAS_CV2 = False

REST_SMILE = 0.15


class AIFaceRenderer:

    def __init__(self, image_path):
        img = Image.open(image_path).convert('RGBA')
        self._orig = np.array(img, dtype=np.float32) / 255.0
        self._orig_h, self._orig_w = self._orig.shape[:2]

        self._detect_mouth()
        self._create_base()

        self._cache_w = -1
        self._cache_h = -1
        self._base_resized = None
        self._line_px = 0
        self._kernel = None
        self._kernel_r = 0

    def _detect_mouth(self):
        gray = np.mean(self._orig[:, :, :3], axis=2)
        h, w = gray.shape

        y0 = int(h * 0.55)
        lower = gray[y0:]
        thresh = 0.3
        mask = lower > thresh

        rows_any = np.any(mask, axis=1)
        cols_any = np.any(mask, axis=0)

        if not np.any(rows_any) or not np.any(cols_any):
            self._m_corner_y = 0.67
            self._m_depth = 0.10
            self._m_cx = 0.50
            self._m_w = 0.50
            self._m_color = np.array([0.86, 0.82, 0.75])
            self._m_line_w = 0.012
            self._erase = (int(h * 0.65), int(h * 0.80), int(w * 0.20), int(w * 0.80))
            return

        row_idx = np.where(rows_any)[0]
        col_idx = np.where(cols_any)[0]

        top = row_idx[0] + y0
        bot = row_idx[-1] + y0
        left = col_idx[0]
        right = col_idx[-1]

        self._m_corner_y = top / h
        self._m_depth = (bot - top) / h
        self._m_cx = (left + right) / 2.0 / w
        self._m_w = (right - left) / w

        region = self._orig[top:bot + 1, left:right + 1]
        region_g = gray[top:bot + 1, left:right + 1]
        bright = region_g > thresh
        if np.any(bright):
            self._m_color = region[bright].mean(axis=0)[:3]
        else:
            self._m_color = np.array([0.86, 0.82, 0.75])

        center_col = (left + right) // 2
        vert = gray[top:bot + 1, center_col]
        bright_v = vert > thresh
        self._m_line_w = max(float(np.sum(bright_v)), 6.0) / h

        self._erase = (top, bot, left, right)

    def _create_base(self):
        self._base = self._orig[:, :, :3].copy()
        t, b, l, r = self._erase
        pad_h = int(self._orig_h * 0.06)
        pad_w = int(self._orig_w * 0.03)
        t = max(0, t - pad_h)
        b = min(self._orig_h, b + pad_h * 2 + 1)
        l = max(0, l - pad_w)
        r = min(self._orig_w, r + pad_w + 1)
        self._base[t:b, l:r] = 0.0

    def _update_cache(self, w, h):
        if self._cache_w == w and self._cache_h == h:
            return

        img = Image.fromarray((self._base * 255).astype(np.uint8))
        img = img.resize((w, h), Image.LANCZOS)
        self._base_resized = np.array(img, dtype=np.float32) / 255.0

        self._line_px = max(3, int(self._m_line_w * h * 0.8))

        r = self._line_px // 2 + 4
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        dist = np.sqrt((xx * xx + yy * yy).astype(np.float32))
        half = self._line_px / 2.0
        core = np.clip(1.0 - (dist - half + 1.0) * 0.5, 0.0, 1.0)
        glow = np.clip(1.0 - dist / (r + 1), 0.0, 1.0) ** 3 * 0.3
        self._kernel = np.maximum(core, glow)
        self._kernel_r = r

        self._cache_w = w
        self._cache_h = h

    def render(self, blendshapes, width, height):
        self._update_cache(width, height)
        frame = self._base_resized.copy()

        jaw = blendshapes.get('jawOpen', 0.0)
        funnel = blendshapes.get('mouthFunnel', 0.0)
        pucker = blendshapes.get('mouthPucker', 0.0)
        stretch = (blendshapes.get('mouthStretch_L', 0.0) +
                   blendshapes.get('mouthStretch_R', 0.0)) / 2.0
        smile = (blendshapes.get('mouthSmile_L', 0.0) +
                 blendshapes.get('mouthSmile_R', 0.0)) / 2.0

        cx = self._m_cx * width
        # Shift mouth down: reduce distance to bottom by 10%
        gap_to_bottom = (1.0 - self._m_corner_y) * height
        corner_y = height - gap_to_bottom * 0.8
        base_depth = self._m_depth * height
        base_w = self._m_w * width * 0.85

        # Animated half-width: stretch widens, funnel/pucker narrows
        roundness = funnel + pucker
        half_w = base_w * 0.5 * (1.0 + stretch * 0.3 - roundness * 0.4)

        # Anchors: left and right mouth corners (shared by both lips)
        p_left = np.array([cx - half_w, corner_y])
        p_right = np.array([cx + half_w, corner_y])

        # Opening: jaw 0-0.3 maps to 0-1.0 (full circular opening)
        open_factor = float(np.clip(jaw * 3.3, 0.0, 1.0))

        # Circle radius: at full open, radius = 55% of half_w
        radius = open_factor * half_w * 0.5

        # Bezier circle approximation: handle offset = (4/3) * radius
        KAPPA = 4.0 / 3.0
        handle_offset = KAPPA * radius

        # Handle X spread: 0.5523 * half_w for circular arc approximation
        hx = half_w * 0.5523

        # Rest smile dip (fades out as mouth opens)
        smile_f = float(np.clip(0.3 + smile * 3.0, 0.3, 1.0))
        rest_dip = base_depth * smile_f * (1.0 - open_factor)

        # Upper lip: dips at rest, rises above corners when open
        upper_y = corner_y + rest_dip - handle_offset
        # Lower lip: overlaps upper at rest, drops below corners when open
        lower_y = corner_y + rest_dip + handle_offset

        upper = _bezier(p_left,
                        np.array([cx - hx, upper_y]),
                        np.array([cx + hx, upper_y]),
                        p_right)
        lower = _bezier(p_left,
                        np.array([cx - hx, lower_y]),
                        np.array([cx + hx, lower_y]),
                        p_right)

        self._draw(frame, upper, 1.0)
        self._draw(frame, lower, 1.0)

        return frame

    def _draw(self, frame, points, alpha):
        if _HAS_CV2:
            self._draw_cv2(frame, points, alpha)
        else:
            self._draw_np(frame, points, alpha)

    def _draw_cv2(self, frame, points, alpha):
        h, w = frame.shape[:2]
        pts = np.clip(points, [0, 0], [w - 1, h - 1]).astype(np.int32).reshape(-1, 1, 2)
        color = tuple(float(c) for c in self._m_color)
        lw = self._line_px

        glow = np.zeros_like(frame)
        cv2.polylines(glow, [pts], False, color, lw + 6, cv2.LINE_AA)
        k = max(5, (lw + 6) | 1)
        glow = cv2.GaussianBlur(glow, (k * 2 + 1, k * 2 + 1), k * 0.6)
        np.maximum(frame, glow * 0.2 * alpha, out=frame)

        a_color = tuple(float(c * alpha) for c in self._m_color)
        cv2.polylines(frame, [pts], False, a_color, lw, cv2.LINE_AA)

    def _draw_np(self, frame, points, alpha):
        kernel = self._kernel * alpha
        r = self._kernel_r
        h, w = frame.shape[:2]
        kh, kw = kernel.shape

        mask = np.zeros((h, w), dtype=np.float32)
        step = max(1, len(points) // 100)
        for i in range(0, len(points), step):
            px = int(round(points[i, 0]))
            py = int(round(points[i, 1]))
            y1 = max(0, py - r)
            y2 = min(h, py + r + 1)
            x1 = max(0, px - r)
            x2 = min(w, px + r + 1)
            if y2 <= y1 or x2 <= x1:
                continue
            ky1 = y1 - (py - r)
            ky2 = kh - ((py + r + 1) - y2)
            kx1 = x1 - (px - r)
            kx2 = kw - ((px + r + 1) - x2)
            np.maximum(mask[y1:y2, x1:x2], kernel[ky1:ky2, kx1:kx2],
                       out=mask[y1:y2, x1:x2])

        for c in range(3):
            np.maximum(frame[:, :, c], mask * self._m_color[c],
                       out=frame[:, :, c])


def _bezier(p0, p1, p2, p3, n=200):
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    t1 = 1.0 - t
    return t1 ** 3 * p0 + 3 * t1 ** 2 * t * p1 + 3 * t1 * t ** 2 * p2 + t ** 3 * p3
