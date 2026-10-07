# Hybrid Images

## Project Description

This project implements a Hybrid Image generation pipeline using
2D cross-correlation, convolution, Gaussian blur, low-pass filtering,
and high-pass filtering.

The final hybrid image is created by combining the low-frequency
information from one image with the high-frequency information from
another image.

The implementation uses Python and NumPy for the filtering operations.
Pillow is used only for image loading, resizing, and saving.
No OpenCV or SciPy filtering functions are used.

No OpenCV or SciPy filtering functions are used.

---

## Implemented Functions

The following five required functions are implemented:

1. `cross_correlation_2d`
2. `convolve_2d`
3. `gaussian_blur_kernel_2d`
4. `low_pass`
5. `high_pass`

### 1. Cross Correlation

`cross_correlation_2d` applies a 2D kernel to the input image
using NumPy operations and explicit loops over image pixels.

Reflect padding is used at the image boundaries.

### 2. Convolution

`convolve_2d` implements convolution by flipping the kernel using
`np.flip` and then applying cross-correlation.

### 3. Gaussian Blur Kernel

`gaussian_blur_kernel_2d` generates a Gaussian kernel with a
user-defined height, width, and sigma.

The kernel is normalized so that its elements sum to 1.

### 4. Low-Pass Filter

The low-pass filter applies Gaussian blur to the image.

This removes high-frequency details while preserving the general
structure of the image.

### 5. High-Pass Filter

The high-pass filter is computed by subtracting the low-pass version
of the image from the original image:

    High-pass = Original Image - Low-pass Image

---

## Filter Parameters

### Low-Pass Filter

- Kernel size: `31 × 31`
- Sigma: `6.0`
- Source image: `left.png`

### High-Pass Filter

- Kernel size: `13 × 13`
- Sigma: `2.0`
- Source image: `right.png`

---

## Hybrid Image

The final hybrid image is generated using:

    Hybrid = Low-pass + 0.5 × High-pass

Therefore, the combination ratio is:

    Low-pass : High-pass = 1 : 0.5

The low-frequency information is taken from `left.png`,
while the high-frequency information is taken from `right.png`.

---

## Image Alignment

The two input images were aligned using the provided GUI tool.

Corresponding facial points were selected in both images so that
the second image could be transformed into the coordinate system
of the first image.

The aligned images are saved as:

- `left.png`
- `right.png`

---

## Output

The final hybrid image is saved as:

    hybrid.png

The final output combines the smooth, low-frequency information
of the cat image with the high-frequency details of the dog image.

---

## Implementation Notes

The filtering operations are implemented without OpenCV or SciPy
filtering functions.

NumPy operations, explicit loops, padding, and kernel operations
are used to implement the required filtering functions.

For faster execution, the input images are resized to a maximum
dimension of 600 pixels before filtering.

---

## How to Run

Place the following files in the same directory:

    hybrid.py
    left.png
    right.png

Then run:

    python hybrid.py

The program generates:

    hybrid.png
