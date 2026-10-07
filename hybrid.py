import numpy as np
from PIL import Image


# ============================================================
# 1. Cross Correlation
# ============================================================

def cross_correlation_2d(image, kernel):
    image = image.astype(np.float64)

    h, w = image.shape[:2]
    kh, kw = kernel.shape[:2]

    pad_h = kh // 2
    pad_w = kw // 2

    if image.ndim == 3:
        padded = np.pad(
            image,
            (
                (pad_h, pad_h),
                (pad_w, pad_w),
                (0, 0)
            ),
            mode="reflect"
        )

        output = np.zeros_like(
            image,
            dtype=np.float64
        )

        for y in range(h):
            for x in range(w):
                region = padded[
                    y:y + kh,
                    x:x + kw,
                    :
                ]

                output[y, x] = np.sum(
                    region * kernel[:, :, np.newaxis],
                    axis=(0, 1)
                )

    else:
        padded = np.pad(
            image,
            (
                (pad_h, pad_h),
                (pad_w, pad_w)
            ),
            mode="reflect"
        )

        output = np.zeros_like(
            image,
            dtype=np.float64
        )

        for y in range(h):
            for x in range(w):
                region = padded[
                    y:y + kh,
                    x:x + kw
                ]

                output[y, x] = np.sum(
                    region * kernel
                )

    return output


# ============================================================
# 2. Convolution
# ============================================================

def convolve_2d(image, kernel):
    flipped_kernel = np.flip(kernel)

    return cross_correlation_2d(
        image,
        flipped_kernel
    )


# ============================================================
# 3. Gaussian Blur Kernel
# ============================================================

def gaussian_blur_kernel_2d(height, width, sigma):

    kernel = np.zeros(
        (height, width),
        dtype=np.float64
    )

    center_y = height // 2
    center_x = width // 2

    for y in range(height):

        for x in range(width):

            dy = y - center_y
            dx = x - center_x

            kernel[y, x] = np.exp(
                -(dx ** 2 + dy ** 2)
                / (2.0 * sigma ** 2)
            )

    # Normalize kernel
    kernel_sum = np.sum(kernel)

    if kernel_sum > 0:
        kernel /= kernel_sum

    return kernel


# ============================================================
# 4. Low Pass
# ============================================================

def low_pass(image, kernel_size, sigma):

    if isinstance(kernel_size, int):

        kh = kernel_size
        kw = kernel_size

    else:

        kh, kw = kernel_size

    kernel = gaussian_blur_kernel_2d(
        kh,
        kw,
        sigma
    )

    return convolve_2d(
        image,
        kernel
    )


# ============================================================
# 5. High Pass
# ============================================================

def high_pass(image, kernel_size, sigma):

    image_float = image.astype(
        np.float64
    )

    low = low_pass(
        image_float,
        kernel_size,
        sigma
    )

    return image_float - low


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    img_left = Image.open(
        "left.png"
    ).convert("RGB")

    img_right = Image.open(
        "right.png"
    ).convert("RGB")


    # --------------------------------------------------------
    # Reduce image size for faster processing
    # --------------------------------------------------------

    MAX_SIZE = 600

    img_left.thumbnail(
        (MAX_SIZE, MAX_SIZE),
        Image.Resampling.LANCZOS
    )

    img_right.thumbnail(
        (MAX_SIZE, MAX_SIZE),
        Image.Resampling.LANCZOS
    )


    # --------------------------------------------------------
    # Make both images the same size
    # --------------------------------------------------------

    if img_left.size != img_right.size:

        img_right = img_right.resize(
            img_left.size,
            Image.Resampling.LANCZOS
        )


    # --------------------------------------------------------
    # Convert images to NumPy arrays
    # --------------------------------------------------------

    left_arr = np.array(
        img_left,
        dtype=np.float64
    )

    right_arr = np.array(
        img_right,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Filter parameters
    # --------------------------------------------------------

    # Low-pass filter
    sigma_low = 6.0
    kernel_size_low = 31


    # High-pass filter
    sigma_high = 2.0
    kernel_size_high = 13


    # --------------------------------------------------------
    # Create Low-pass image
    # --------------------------------------------------------

    print("Processing low-pass...")

    low = low_pass(
        left_arr,
        kernel_size_low,
        sigma_low
    )


    # --------------------------------------------------------
    # Create High-pass image
    # --------------------------------------------------------

    print("Processing high-pass...")

    high = high_pass(
        right_arr,
        kernel_size_high,
        sigma_high
    )


    # --------------------------------------------------------
    # Create Hybrid Image
    # --------------------------------------------------------

    print("Creating hybrid...")

    # Hybrid =
    # Low-pass image + 0.5 * High-pass image

    hybrid = (
        low
        + 0.5 * high
    )


    # --------------------------------------------------------
    # Convert to valid image range
    # --------------------------------------------------------

    hybrid = np.clip(
        hybrid,
        0,
        255
    ).astype(np.uint8)


    # --------------------------------------------------------
    # Save final Hybrid Image
    # --------------------------------------------------------

    Image.fromarray(
        hybrid
    ).save(
        "hybrid.png"
    )


    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("==============================")
    print("DONE!")
    print("hybrid.png created successfully.")
    print("==============================")