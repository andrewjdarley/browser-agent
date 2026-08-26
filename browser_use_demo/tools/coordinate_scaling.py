"""
Coordinate scaling utilities for browser tool.

This module handles the scaling of coordinates from Claude's vision model
resolution to the actual browser viewport resolution.
"""


class CoordinateScaler:
    """Handles coordinate scaling between Claude's vision and actual viewport."""

    # Claude resizes oversized images before seeing them, preserving aspect ratio -
    # for a 16:9 screenshot it lands on exactly these dimensions:
    # https://docs.claude.com/en/docs/build-with-claude/vision#evaluate-image-size
    CLAUDE_ACTUAL_WIDTH = 1456
    CLAUDE_ACTUAL_HEIGHT = 819

    @classmethod
    def get_scale_factors(cls, viewport_width: int, viewport_height: int) -> tuple[float, float]:
        """Calculate scale factors for converting Claude coordinates to viewport coordinates."""
        scale_x = viewport_width / cls.CLAUDE_ACTUAL_WIDTH
        scale_y = viewport_height / cls.CLAUDE_ACTUAL_HEIGHT
        return scale_x, scale_y

    @classmethod
    def scale_coordinates(
        cls,
        x: int,
        y: int,
        viewport_width: int,
        viewport_height: int,
        apply_threshold: bool = True
    ) -> tuple[int, int]:
        """Scale an (x, y) pair from Claude's vision resolution to actual
        viewport pixels. apply_threshold guards against double-scaling a
        coordinate that's already in viewport space (larger than Claude's
        resolution could produce) by passing it through unchanged."""
        scale_x, scale_y = cls.get_scale_factors(viewport_width, viewport_height)

        if abs(scale_x - 1.0) < 0.05 and abs(scale_y - 1.0) < 0.05:
            return x, y

        if apply_threshold:
            max_expected_x = cls.CLAUDE_ACTUAL_WIDTH * 1.2
            max_expected_y = cls.CLAUDE_ACTUAL_HEIGHT * 1.2
            if x > max_expected_x or y > max_expected_y:
                return x, y

        scaled_x = min(int(x * scale_x), viewport_width - 1)
        scaled_y = min(int(y * scale_y), viewport_height - 1)
        return scaled_x, scaled_y

    @classmethod
    def scale_coordinate_list(
        cls,
        coords: list | tuple,
        viewport_width: int,
        viewport_height: int
    ) -> list:
        """Scale a [x, y] coordinate pair."""
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            return list(coords) if isinstance(coords, tuple) else coords

        x, y = coords[0], coords[1]
        scaled_x, scaled_y = cls.scale_coordinates(x, y, viewport_width, viewport_height)
        return [scaled_x, scaled_y]