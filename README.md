# Human Pose Estimation Streamlit Project

This project integrates human pose estimation using Streamlit as a web interface, leveraging AlphaPose for 2D pose detection and MotionBERT for 3D pose estimation and biomechanics analysis.

## Project Structure

- **streamlit.py**: Main Streamlit app for user interaction and visualization.
- **AlphaPose/**: Contains the AlphaPose framework for 2D human pose estimation.
- **MotionBERT/**: Contains MotionBERT scripts and pretrained models for 3D pose estimation and biomechanics extraction.
- **tempfiles/**: Temporary files and intermediate outputs.

## Features

- Upload videos or images for pose estimation.
- 2D pose detection using AlphaPose.
- 3D pose estimation and biomechanics analysis using MotionBERT.
- Visualization of results in the Streamlit web app.

## Getting Started

1. Clone the repository and install dependencies for both AlphaPose and MotionBERT.
2. Run the Streamlit app:
   ```bash
   streamlit run streamlit.py
   ```
3. Follow the web interface to upload files and view results.

## Notable Script Modifications and Contributions

- **AlphaPose/scripts/demo_inference.py**: Slightly modified to allow running on CPU instead of only GPU. This enables pose estimation on systems without CUDA support.
- **MotionBERT/infer_wild_2.py**: Rewritten from the original script, with the addition of temporal smoothing for improved 3D pose estimation results.
- **MotionBERT/extract_biomechanics_single.py**: Script developed by us to extract cyclist biomechanics from 3D pose data.
- **streamlit.py**: Fully implemented by us to provide an interactive web interface for testing and using the entire pose estimation and biomechanics pipeline.

## Credits

- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
- [MotionBERT](https://github.com/DeepMotionEditing/MotionBERT)

---

Feel free to update this README with more details about your workflow, results, or instructions as needed.
