import numpy as np
import pytest

from albumentations.augmentations.domain_adaptation import apply_histogram


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("reference_image_dtype", [np.uint8, None])
def test_apply_histogram_dtype_conversion(dtype, reference_image_dtype):
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)

    # `None` is used to set the same dtype for `reference_image` and `img`
    reference_image = np.array(
        [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
        dtype=reference_image_dtype if reference_image_dtype is not None else np.uint8,
    )

    blend_ratio = 0.5
    output = apply_histogram(img, reference_image, blend_ratio)
    expected = np.array([[2, 2, 2], [5, 5, 5], [8, 8, 8]], dtype=dtype)
    assert output.dtype == expected.dtype
    assert np.all(output == expected)
