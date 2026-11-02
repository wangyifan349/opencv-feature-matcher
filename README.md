# 🔎 OpenCV Feature Matcher

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-feature%20matching-5C3EE8?logo=opencv&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-2E8B57.svg)

**Find a known object in another image with classic OpenCV features.** This standalone Tkinter app draws feature-match connections and, when the geometry checks out, outlines the reference image's projected location in the scene.

> Everything runs locally. No model training, account, or image upload is needed.

## ⚡ Quick start

```bash
python -m pip install opencv-contrib-python numpy
python opencv_feature_matcher.py
```

Requires **Python 3.10+** and Tkinter (usually included with standard Python on Windows and macOS).

1. Load a close, clear view of the object as the **Reference Image**.
2. Load the larger image to search as the **Scene Image**.
3. Choose **ORB**, **SIFT**, or **AKAZE**, then click **Find Match**.
4. Review the match visualization. Click **Save Result Image** to export it as JPEG, PNG, or BMP.

## ✨ What it does

- Detects keypoints and computes descriptors with **ORB**, **SIFT**, or **AKAZE**.
- Uses K-nearest-neighbor descriptor matching and Lowe's ratio test to filter ambiguous candidates.
- Estimates a perspective transform (homography) with **RANSAC** to identify geometrically consistent matches.
- Draws match connections side by side. When a homography is found, the connections show RANSAC inliers; otherwise, ratio-filtered candidates are shown.
- Draws a **yellow quadrilateral** only when the homography has at least the configured number of inliers. It projects the four corners of the *entire reference image*—it is not a segmentation mask.
- Optionally applies CLAHE local contrast enhancement before feature detection.
- Runs matching on a background thread so the interface remains responsive.
- Includes an **About Algorithms** tab with an overview of the program and its core pipeline.

## 🧭 Choose a feature method

| Method | Descriptor / distance | Good starting point | Main trade-off |
| --- | --- | --- | --- |
| **ORB** | Binary / Hamming | Fast general-purpose matching | Very fast; may be less robust to large scale or viewpoint changes |
| **SIFT** | Floating-point / L2 | Scale or rotation changes | Often more robust; typically uses more computation |
| **AKAZE** | Binary MLDB / Hamming | Nonlinear scale-space features | A useful alternative to ORB; detection speed depends on the image and settings |

There is no universally best method. Try ORB first; compare SIFT or AKAZE if the images differ significantly in scale, rotation, or appearance.

## 🙋 About

This project is a hands-on desktop tool for exploring classical feature matching. It makes the main stages visible—from local keypoints and descriptor matches to RANSAC verification and the projected reference boundary. It is intended for experimentation and planar-object localization, not as a general-purpose semantic object detector.

## 🎛️ Parameters

The **Algorithm Parameters** panel changes with the selected method:

| Method | Parameters |
| --- | --- |
| **ORB** | **Max Features** limits detected keypoints; **Scale Factor** and **Pyramid Levels** set the image scale pyramid; **FAST Threshold** controls detector sensitivity. |
| **SIFT** | **Max Features** limits detected keypoints; **Contrast Threshold** filters weak keypoints; **Edge Threshold** filters edge-like responses; **Sigma** sets the initial blur. |
| **AKAZE** | **Detection Threshold** controls detector sensitivity; **Octaves** and **Layers per Octave** set the nonlinear scale space; **Diffusivity** selects its diffusion model. |

Shared controls:

- **Lowe Match Ratio** (default `0.75`): lower is stricter and usually keeps fewer, cleaner candidates.
- **Min. RANSAC Inliers** (default `10`): minimum geometrically consistent matches needed before the yellow outline is drawn.
- **Enhance contrast (CLAHE)**: optional local contrast enhancement; it can help low-contrast images but may also amplify noise.

## 🧠 How localization works

1. Convert both images to grayscale; optionally apply CLAHE.
2. Detect keypoints and compute descriptors with the selected feature method.
3. Match each reference descriptor to its two nearest scene descriptors and apply the Lowe ratio test.
4. Estimate a homography using RANSAC (a fixed 5-pixel reprojection threshold in the current implementation).
5. Count RANSAC inliers and project the reference image's four corners into the scene. Draw the yellow outline only if the inlier count meets **Min. RANSAC Inliers**.

This is **feature-based geometric localization**, not semantic object recognition. It works best when the target is textured, in focus, mostly planar, and not heavily occluded. Repeated patterns, blur, large perspective changes, or major appearance differences can produce false matches or no detection. A tight reference crop helps ensure the outline corresponds to the object.

## 📚 More detail

See the [OpenCV Feature Matcher Guide](OPENCV_FEATURE_MATCHER.md) for the full workflow, algorithm notes, and troubleshooting tips.

## ☕ Support the developer

If you’d like to support the developer, consider buying me a coffee with Bitcoin. Any support is greatly appreciated:

`bc1q7ye9xq0gcp2z9dh5lzafgld6z29nzt5m68tsfj`

## 📄 License

Released under the permissive [MIT License](LICENSE). Third-party dependencies retain their own licenses.
