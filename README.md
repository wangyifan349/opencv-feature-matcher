# OpenCV Feature Matcher

A standalone desktop application for locating a reference object in a scene image. It uses OpenCV feature detection, descriptor matching, and RANSAC-based geometric verification. The Tkinter interface and all project documentation are in English.

## Features

- Select **ORB**, **SIFT**, or **AKAZE** with method-specific settings.
- Filter ambiguous matches with KNN matching and the Lowe ratio test.
- Estimate a perspective homography with RANSAC, draw inlier match lines, and outline a detected object.
- Optionally apply CLAHE local contrast enhancement.
- Keep the interface responsive by running matching in a background thread.
- Save the match visualization as a JPEG, PNG, or BMP image.
- Learn about the program and algorithms in the **About Algorithms** tab.

## Requirements

- Python 3.10 or newer
- Tkinter (included with most standard Python installations)
- OpenCV and NumPy

Install the Python packages and launch the application from this directory:

```bash
python -m pip install opencv-contrib-python numpy
python opencv_feature_matcher.py
```

## Quick start

1. Load a clear close-up of the object as the **Reference Image**.
2. Load the larger image to search as the **Scene Image**.
3. Choose an algorithm and click **Find Match**. ORB is a fast default; SIFT can be more robust to scale changes; AKAZE is another binary-feature option.
4. Green lines show geometrically consistent feature matches. A yellow outline appears when RANSAC finds enough inliers to locate the object.
5. Click **Save Result Image** to export the visualization. Results can be saved even if a reliable outline was not found.

Algorithm-specific parameters appear when you change the selected method. Lowe Match Ratio, Min. RANSAC Inliers, and optional CLAHE are shared settings. The **About Algorithms** tab describes the core pipeline and settings.

For more detail, see [OPENCV_FEATURE_MATCHER.md](OPENCV_FEATURE_MATCHER.md).

## Support the developer

If you’d like to support the developer, consider buying me a coffee with Bitcoin. Any support is greatly appreciated:

**Bitcoin address:** `bc1q7ye9xq0gcp2z9dh5lzafgld6z29nzt5m68tsfj`

## License

This project is released under the permissive [MIT License](LICENSE). Third-party dependencies retain their own licenses.
