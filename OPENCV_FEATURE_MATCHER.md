# 📘 OpenCV Feature Matcher — User Guide

This guide explains the app's workflow, feature methods, matching pipeline, and the meaning of the result overlay.

## 🙋 About this application

The app is a compact, local desktop tool for learning and experimenting with classical OpenCV feature matching. It lets you compare ORB, SIFT, and AKAZE on your own image pairs and inspect how descriptor filtering and geometric verification affect localization.

## 🖥️ The Matching tab

The Matching tab contains the image loaders, algorithm selector, method-specific parameters, shared matching controls, previews, and result image.

1. **Load Reference Image** — choose a clear crop showing the object you want to find.
2. **Load Scene Image** — choose the larger image in which to look for it.
3. **Feature Method** — select ORB, SIFT, or AKAZE. The parameter panel updates for that method.
4. **Find Match** — starts matching in a worker thread. The window remains responsive while OpenCV processes the images.
5. **Save Result Image** — exports the displayed side-by-side visualization as JPEG, PNG, or BMP.

## 🧩 Understanding the result

- **Green lines** connect matching keypoints. If RANSAC produced a homography, only RANSAC inliers are drawn. If it could not estimate one, the app displays the Lowe-ratio-filtered candidate matches instead.
- **Yellow quadrilateral** appears only when a homography was found and its inlier count meets **Min. RANSAC Inliers**. It is the projected boundary of the entire reference image, not a pixel-accurate object mask.
- The scene image is placed to the right of the reference image in the saved match visualization. The outline is offset accordingly.

You can save the visualization even when no yellow outline appears. That can be useful for inspecting candidate matches or tuning parameters.

## ⚙️ Feature methods and their parameters

### ORB

ORB combines FAST keypoint detection with a binary descriptor. It is a fast default and uses Hamming distance.

- **Max Features** — maximum requested number of keypoints (default: `1500`).
- **Scale Factor** — scale change between pyramid levels (default: `1.2`). Values must be greater than `1.0`.
- **Pyramid Levels** — number of scale-pyramid levels (default: `8`).
- **FAST Threshold** — detector threshold (default: `20`); lower values can detect more points, including weaker ones.

### SIFT

SIFT computes floating-point descriptors and uses L2 distance. Its scale-space design is often useful when the object appears at a different size or rotation, with more computation than ORB.

- **Max Features** — maximum requested number of keypoints (default: `1500`).
- **Contrast Threshold** — rejects low-contrast extrema (default: `0.04`); lowering it generally retains more keypoints.
- **Edge Threshold** — filters edge-like responses (default: `10`). In OpenCV SIFT, a larger value retains more edge responses.
- **Sigma** — assumed blur of the input image at the first octave (default: `1.6`).

### AKAZE

AKAZE detects features in nonlinear scale space. The app uses OpenCV's default MLDB binary descriptor, which is matched with Hamming distance.

- **Detection Threshold** — detector response threshold (default: `0.001`); lowering it generally detects more keypoints.
- **Octaves** — number of nonlinear scale-space octaves (default: `4`).
- **Layers per Octave** — number of evolution steps in each octave (default: `4`).
- **Diffusivity** — selects the nonlinear diffusion model (default: `PM_G2`).

### Shared controls

- **Lowe Match Ratio** (`0.75` by default) compares the best and second-best descriptor distances. Lower values reject more ambiguous matches; higher values admit more candidates and may increase false matches.
- **Min. RANSAC Inliers** (`10` by default) is the minimum support required to draw the yellow outline. It does not change the feature detector or the RANSAC fit itself.
- **Enhance contrast (CLAHE)** optionally applies CLAHE to both grayscale images before feature detection, using a clip limit of `2.0` and an `8 × 8` tile grid. It can help uneven or low contrast, but may amplify noise.

## 🔬 Matching pipeline

1. Decode both images to color, then convert them to grayscale.
2. Optionally enhance local contrast with CLAHE.
3. Detect keypoints and compute descriptors with the selected method.
4. Use OpenCV's brute-force matcher to find the two nearest scene descriptors for each reference descriptor.
5. Apply Lowe's ratio test to discard ambiguous candidates.
6. With at least four candidates, estimate a perspective homography using `cv2.findHomography` with RANSAC and a `5.0` pixel reprojection threshold.
7. Draw RANSAC inliers when a homography is available; otherwise draw the ratio-filtered candidates. If the homography has enough inliers, transform the four reference-image corners and draw the yellow polygon.

## 💡 Practical tips and limitations

- Use a sharp reference crop with distinctive texture or corners. Avoid including large amounts of unrelated background around the target.
- This technique is best suited to planar objects (such as a book cover, sign, label, or poster). A single homography may not describe a strongly three-dimensional object.
- Repeated patterns can create convincing but incorrect matches. Blur, occlusion, lighting changes, extreme viewpoint changes, and very small targets can prevent reliable localization.
- Try ORB first for speed. Try SIFT if scale or rotation changes are challenging, or AKAZE as another binary-descriptor option.
- Adjust one parameter at a time. If there are too few candidates, a slightly higher Lowe ratio or lower detector threshold may help; if there are many false matches, try a lower Lowe ratio or a higher minimum inlier count.

## 🪪 License

The application and its documentation are released under the [MIT License](LICENSE). OpenCV, NumPy, and other third-party packages remain subject to their own licenses.
