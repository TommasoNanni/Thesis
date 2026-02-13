from pathlib import Path
import numpy as np

from data.video_dataset import EgoExoSceneDataset
from Thesis.data.parameters_extraction import ParameterEstimator


def main():
    data_root = "/cluster/project/cvg/data/EgoExo_georgiatech/raw/takes"
    output_dir = "/cluster/project/cvg/students/tnanni/Thesis/test_outputs/sam3d_test"

    ds = EgoExoSceneDataset(data_root, slice=1)
    segmenter = ParameterEstimator(smooth=False)
    results = segmenter.segment_scene_extract_parameters(
        ds[0], output_dir, vis=True, match_across_videos=True,
    )

    for video_id, video_dir in results.items():
        body_dir = Path(video_dir) / "body_data"
        if not body_dir.exists():
            print(f"  WARNING: {body_dir} does not exist")
            continue

        npz_files = sorted(body_dir.glob("person_*.npz"))
        print(f"\n--- {video_id}: {len(npz_files)} person files ---")
        for npz_path in npz_files:
            data = dict(np.load(str(npz_path)))
            print(f"  {npz_path.name}:")
            for key, arr in data.items():
                print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")

        summary_path = body_dir / "body_params_summary.json"
        if summary_path.exists():
            import json
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"  Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
