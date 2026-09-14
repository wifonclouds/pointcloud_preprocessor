# PointCloud Preprocessor

Simple preprocessing pipeline for indoor point clouds.

Pipeline:

1. Load point cloud
2. Translate to origin
3. Rotate to global axes
4. Estimate normals
5. Detect floor
6. Detect walls
7. Crop above walls
8. Save results

step1: mkdir -p data/"NEW SITE"/{input,output,control_points}
step2: place the raw point cloud in the input folder, place the control points (ref, 2 point angle) in control_points
step3: python main.py "NEW SITE"
