# pyNanoFind

This package provides the code to develop and use a convolutional neural network
for semantic segmentation of high resolution transmission electron microscopy
(HRTEM) micrographs and classify stacking faults. The provided network is for micrographs with size 512x512.

Example data preprocessing workflow is demonstrated in the notebook in "data
preprocessing" directory.

Setup for network training can be found in training files.

Example implementation of the segmentation network can be found in segmentation demo.

Setup for stacking fault classification and demonstration can be found in random forest demo.

Bragg filtering provides a workflow to implement bragg peak filtering for a
comparison segmentation method.
