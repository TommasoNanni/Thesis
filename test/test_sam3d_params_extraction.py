"""
Test only for checking that the parameters are extracted correctly and SAM3D loads well
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

import json
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn

from data.video_dataset import EgoExoSceneDataset
from data.segmentation import PersonSegmenter
from data.parameters_extraction import BodyParameterEstimator

def main():
    data_root = "/cluster/project/cvg/data/EgoExo_georgiatech/raw/takes"
    output_dir = "/cluster/project/cvg/students/tnanni/ghost/test_outputs/segmentation_test"

    # Load dataset (1 scene for testing)
    ds = EgoExoSceneDataset(data_root, slice=1)
    scene = ds[0]
    print(f"\n=== Scene: {scene.scene_id} ({len(scene)} videos) ===")
    for v in scene:
        print(f"  {v}")

    # Step 1: Segment people in the scene
    segmenter = PersonSegmenter()
    print(f"\n--- Running segmentation on scene '{scene.scene_id}' ---")
    video_dirs = segmenter.segment_scene(
        scene=scene,
        output_dir=output_dir,
        vis=False,
    )
    print(f"\nSegmentation output dirs:")
    for video_id, vdir in video_dirs.items():
        print(f"  {video_id}: {vdir}")

    # Step 2: Estimate body parameters from segmentation output
    estimator = BodyParameterEstimator()
    print(f"\n--- Running body parameter estimation ---")
    estimator.estimate_scene(
        scene=scene,
        video_dirs=video_dirs,
    )


if __name__ == "__main__":
    main()
