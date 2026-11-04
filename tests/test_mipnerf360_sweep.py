"""The Mip-NeRF 360 sweep's per-scene resolution, checked without a GPU.

3DGS trains outdoor scenes on images_4 and indoor scenes on images_2. A scene
filed under the wrong list trains at twice or half the resolution its published
numbers used, and nothing downstream would notice.
"""

import pytest

from bench.mipnerf360_sweep import INDOOR, OUTDOOR, SCENES, image_dir


def test_outdoor_scenes_use_a_quarter_resolution_and_indoor_half():
    assert {s: image_dir(s) for s in SCENES} == {
        "bicycle": "images_4", "flowers": "images_4", "garden": "images_4",
        "stump": "images_4", "treehill": "images_4",
        "room": "images_2", "counter": "images_2", "kitchen": "images_2",
        "bonsai": "images_2"}


def test_the_nine_scenes_are_each_listed_once():
    assert len(SCENES) == len(set(SCENES)) == 9
    assert not set(OUTDOOR) & set(INDOOR)


def test_an_unknown_scene_is_refused_before_training():
    with pytest.raises(ValueError, match="unknown Mip-NeRF 360 scene"):
        image_dir("gardn")
