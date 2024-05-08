import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import streamlit as st
import click
from PIL import Image
import time
import itertools as it
import simpose as sp

from streamlit_image_comparison import image_comparison
import tensorflow as tf


@click.command()
@click.argument("data_dir", type=click.Path(exists=True, path_type=Path))
def main(data_dir: Path) -> None:
    st.set_page_config(layout="wide", page_title="Dataset Viewer")

    if "last_idx" not in st.session_state:
        st.session_state["last_idx"] = 0

    c1, c2, c3 = st.columns(3)

    # c1
    img_dir = c1.text_input(
        "Image directory",
        str(data_dir.resolve()),
        label_visibility="collapsed",
        placeholder="Dataset directory",
    )
    indices = get_idx(Path(img_dir))

    if len(indices) == 1:
        indices += [indices[0]]
    idx = c1.select_slider("Select image", indices, value=indices[0], key="idx")

    # c2
    if c2.button("↻"):
        st.cache_data.clear()

    # c3
    with c3:
        use_bbox = st.toggle("BBox", value=False)
        use_pose = st.toggle("Pose", value=False)

    st.header(f"Datapoint: #{idx:05} (of total {len(indices)} images)")

    assert isinstance(idx, np.int64), f"Got {type(idx)} instead"

    data = get_data(Path(img_dir), idx, use_bbox=use_bbox, use_pose=use_pose)

    chose_col1, chose_col2, _ = st.columns(3)
    chosen_left = chose_col1.selectbox(
        "Select view",
        options=list(data.keys()),
        index=0,
        key="view_left",
    )
    chosen_right = chose_col2.selectbox(
        "Select view",
        options=list(data.keys()),
        index=1,
        key="view_right",
    )

    left_img = data[chosen_left]
    right_img = data[chosen_right]

    # create preview, with rgb and mask
    c1, c2 = st.columns(2)
    with c1:
        image_comparison(
            img1=Image.fromarray(left_img),
            img2=Image.fromarray(right_img),
            label1=chosen_left,
            label2=chosen_right,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True,
        )
    with c2:
        raw_data = load_data(Path(img_dir), idx)
        st.markdown("Available keys in the dataset")
        st.table({k: f"{str(d.dtype)};{d.shape}" for k, d in raw_data.items()})

    cols = it.cycle(st.columns(3))
    for key, img in data.items():
        with next(cols):
            st.image(img, caption=f"{key} {img.shape}, {img.dtype}", use_column_width=True)


@st.cache_data()
def get_idx(img_dir: Path) -> np.ndarray | list[int]:

    # ---- composite subsets dataset
    if img_dir.joinpath("subsets").exists():
        num = 0
        for subset in img_dir.joinpath("subsets").iterdir():
            end_indices = [
                int(x.stem.split(".")[0].split("_")[-1])
                for x in subset.joinpath("data").glob("*.tfrecord")
            ]
            end_index = max(end_indices)
            num += end_index + 1
        return np.arange(num)

    # ---- single tfrecord dataset
    tfrecord_dir = (
        img_dir.joinpath("data") if img_dir.joinpath("data").exists() else img_dir.joinpath("rgb")
    )

    if len(list(tfrecord_dir.glob("*.tfrecord"))) > 0:
        # tfrecord dataset
        end_indices = [
            int(x.stem.split(".")[0].split("_")[-1]) for x in tfrecord_dir.glob("*.tfrecord")
        ]
        end_index = max(end_indices)
        return np.arange(end_index + 1)
    raise ValueError("No tfrecord found.")


@st.cache_data()
def load_data(img_dir: Path, idx: np.int64) -> dict[str, np.ndarray]:
    tfds = sp.data.TFRecordDataset.get(img_dir, num_parallel_files=1)
    data = tfds.skip(idx).take(1).get_single_element()
    return data


def get_data(
    img_dir: Path, idx: np.int64, use_bbox: bool = False, use_pose: bool = False
) -> dict[str, np.ndarray]:

    data = load_data(img_dir, idx)

    if data is None:
        raise RuntimeError("Could not find data point.")

    sp_keys = sp.data.Dataset

    output = {}
    if sp_keys.RGB in data:
        output["RGB"] = data[sp_keys.RGB].numpy()

    if sp_keys.RGB_R in data:
        output["RGB_R"] = data[sp_keys.RGB_R].numpy()

    if sp_keys.DEPTH in data:
        depth = data["depth"].numpy()
        output["Depth"] = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=255 / np.max(depth)), cv2.COLORMAP_JET  # type: ignore
        )

    if sp_keys.DEPTH_R in data:
        depth_R = data["depth_R"].numpy()
        output["Depth_R"] = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_R, alpha=255 / np.max(depth_R)), cv2.COLORMAP_JET  # type: ignore
        )

    if sp_keys.DEPTH_GT in data:
        depth_gt = data["depth_gt"].numpy()
        output["Depth GT"] = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_gt, alpha=255 / np.max(depth_gt)), cv2.COLORMAP_JET  # type: ignore
        )

    if sp_keys.DEPTH_GT_R in data:
        depth_gt_R = data["depth_gt_R"].numpy()
        output["Depth GT_R"] = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_gt_R, alpha=255 / np.max(depth_gt_R)), cv2.COLORMAP_JET  # type: ignore
        )

    mask = None
    if sp_keys.MASK in data:
        mask = data["mask"].numpy()
        mask_scaled = cv2.convertScaleAbs(mask, alpha=255.0 / np.max(mask))
        colored_mask_bgr = cv2.applyColorMap(mask_scaled, cv2.COLORMAP_TURBO)
        output["Instance Mask"] = cv2.cvtColor(colored_mask_bgr, cv2.COLOR_BGR2RGB)

    if sp_keys.OBJ_IDS in data and sp_keys.OBJ_CLASSES in data and mask is not None:
        cls_ids = [
            {"class": cls, "obj_id": obj_id}
            for cls, obj_id in zip(
                data[sp_keys.OBJ_CLASSES].numpy(), data[sp_keys.OBJ_IDS].numpy()
            )
        ]
        colored_semantic_mask_bgr = assemble_semantic_mask(mask, cls_ids)
        output["Semantic Mask"] = cv2.cvtColor(colored_semantic_mask_bgr, cv2.COLOR_BGR2RGB)

    if use_bbox and sp_keys.OBJ_BBOX_VISIB in data:
        output["RGB"] = draw_bboxes(
            output["RGB"], data[sp_keys.OBJ_BBOX_VISIB].numpy(), data[sp_keys.OBJ_CLASSES].numpy()
        )

    if (
        use_pose
        and sp_keys.OBJ_POS in data
        and sp_keys.OBJ_ROT in data
        and sp_keys.CAM_LOCATION in data
        and sp_keys.CAM_ROTATION in data
        and sp_keys.CAM_MATRIX in data
    ):
        cam_matrix = data[sp_keys.CAM_MATRIX].numpy()
        cam_rot = R.from_quat(data[sp_keys.CAM_ROTATION].numpy()).as_matrix()
        cam_pos = data[sp_keys.CAM_LOCATION].numpy().squeeze()

        obj_poss = data[sp_keys.OBJ_POS].numpy()
        obj_rots = data[sp_keys.OBJ_ROT].numpy()

        output["RGB"] = draw_poses(
            output["RGB"],
            obj_poss,
            obj_rots,
            cam_pos,
            cam_rot,
            cam_matrix,
        )

    return output


def assemble_semantic_mask(mask, objs_data) -> np.ndarray:
    colored_semantic_mask_bgr = np.zeros((*mask.shape[:2], 3)).astype(np.uint8)

    cls_colors = st.session_state["cls_colors"]

    for obj_data in objs_data:
        # semantic
        cls = obj_data["class"]
        cls_colors.setdefault(cls, np.random.randint(0, 256, size=3).astype(np.uint8).tolist())
        colored_semantic_mask_bgr[mask == obj_data["obj_id"]] = cls_colors[cls]

    return colored_semantic_mask_bgr


def draw_poses(img, positions, rotations, cam_pos, cam_rot, cam_matrix) -> np.ndarray:
    for obj_pos, rot_q in zip(positions, rotations):
        obj_rot = R.from_quat(rot_q).as_matrix()

        t = (obj_pos - cam_pos) @ cam_rot
        RotM = cam_rot.T @ obj_rot

        rotV, _ = cv2.Rodrigues(RotM)
        img = cv2.drawFrameAxes(
            img,
            cameraMatrix=cam_matrix,
            rvec=rotV,
            tvec=t,
            distCoeffs=np.zeros(5),
            length=0.05,
            thickness=2,
        )

    return img


def draw_bboxes(img, boxes, classes) -> np.ndarray:
    cls_colors = st.session_state["cls_colors"]

    for bbox, cls in zip(boxes, classes):
        cv2.rectangle(
            img,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color=cls_colors[cls],
            thickness=2,
        )
        # write class name
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            img,
            str(cls),
            (bbox[0], bbox[1]),
            font,
            1,
            cls_colors[cls],
            2,
            cv2.LINE_AA,
        )
    return img


if __name__ == "__main__":
    if "cls_colors" not in st.session_state:
        st.session_state["cls_colors"] = {}
    main()
