"""Non-overlapping, balanced image tiles for bounded attention memory."""

import math


def split_image(inputs, maximum_size=3584):
    height, width = inputs.shape[-2:]
    if maximum_size < 32:
        raise ValueError('Maximum tile size must be at least 32')
    rows = math.ceil(height / maximum_size)
    cols = math.ceil(width / maximum_size)
    # Integer boundaries cover all pixels without overlap or narrow edge tiles.
    return [inputs[:, :, row * height // rows:(row + 1) * height // rows,
                   col * width // cols:(col + 1) * width // cols]
            for row in range(rows) for col in range(cols)]
