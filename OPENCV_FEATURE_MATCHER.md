# OpenCV Feature Matcher Guide

The OpenCV Feature Matcher is a Tkinter desktop application for finding a reference object inside a scene image. It supports ORB, SIFT, and AKAZE feature algorithms and uses OpenCV for descriptor matching and geometric verification.

## Installation and launch

Requirements: Python 3.10 or newer, Tkinter, OpenCV, and NumPy. Tkinter is included with most standard Python installations.

```bash
python -m pip install opencv-contrib-python numpy
python orb_image_matcher.py
```

## Matching workflow

1. Load a clear close-up of the object as the **Reference Image**.
2. Load the larger image to search as the **Scene Image**.
3. Select a feature method and adjust its algorithm-specific settings if needed. ORB is a fast default; SIFT can be more robust to scale changes; AKAZE is an alternative binary-feature method.
4. Click **Find Match**. Green lines show geometrically consistent feature matches. If RANSAC finds enough inliers, a yellow polygon marks the reference image projected into the scene.
5. Click **Save Result Image** to save the visualization, including matches even when a reliable object outline was not found.

## Core algorithm

1. Convert both input images to grayscale. CLAHE can optionally enhance local contrast.
2. Detect keypoints and compute descriptors using ORB, SIFT, or AKAZE.
3. Find two nearest descriptor neighbors with KNN matching, then apply the Lowe ratio test to filter ambiguous matches.
4. Estimate a perspective homography with RANSAC and keep geometrically consistent matches as inliers.
5. Draw the inlier connections and project the reference image corners into the scene to outline the detected object.

## Algorithm-specific settings

- **ORB:** Max Features, Scale Factor, Pyramid Levels, and FAST Threshold. ORB is fast and uses binary descriptors with Hamming distance.
- **SIFT:** Max Features, Contrast Threshold, Edge Threshold, and Sigma. SIFT uses floating-point descriptors and L2 distance, and is often more robust to scale and rotation changes.
- **AKAZE:** Detection Threshold, Octaves, Layers per Octave, and Diffusivity. AKAZE uses nonlinear scale space and binary descriptors with Hamming distance.
- **Shared settings:** Lowe Match Ratio controls match strictness; Min. RANSAC Inliers sets the minimum support needed to draw the object outline; CLAHE is optional local contrast enhancement.

## Tips and limitations

Use a textured, in-focus reference image and a scene where the object is visible. This method works best for mostly planar objects. Repeated patterns, blur, occlusion, major perspective changes, or large appearance differences can cause incorrect or missing detections. Matching runs in a background thread so the interface stays responsive.
