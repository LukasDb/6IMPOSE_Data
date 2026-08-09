import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import streamlit as st
import click
from PIL import Image
import itertools as it
import simpose as sp

from streamlit_image_comparison import image_comparison
import tensorflow as tf


@click.command()
@click.argument("data_dir", type=click.Path(exists=True, path_type=Path))
def main(data_dir: Path) -> None:
    st.set_page_config(layout="wide", page_title="Dataset Viewer")

    with st.sidebar:
        img_dir = st.text_input(
            "Image directory",
            str(data_dir.resolve()),
            label_visibility="collapsed",
            placeholder="Dataset directory",
        )
        c1, c2, c3 = st.columns(3)
        if c1.button("Prev"):
            st.session_state["index"] -= 1
        if c2.button("Next"):
            st.session_state["index"] += 1

        if c3.button("↻"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state["index"] = 0

        use_bbox = st.toggle("BBox", value=False)
        use_pose = st.toggle("Pose", value=False)
        shuffle = st.toggle("Shuffle", value=False)

    # (img_dir, shuffle) identifies the dataset iterator; when it changes,
    # get_ds hands out a fresh one, so restart the index here. Doing it in
    # get_ds would be a side effect inside a cached function.
    ds_key = (img_dir, shuffle)
    if st.session_state.get("_ds_key") != ds_key:
        st.session_state["_ds_key"] = ds_key
        st.session_state["index"] = 0

    idx = st.session_state["index"]
    st.header(f"Datapoint: #{idx:05}")

    data = get_data(Path(img_dir), idx, use_bbox=use_bbox, use_pose=use_pose, shuffle=shuffle)

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
        raw_data = load_data(Path(img_dir), idx, shuffle=shuffle)
        st.markdown("Available keys in the dataset")
        st.table({k: f"{str(d.dtype)};{d.shape}" for k, d in raw_data.items()})

    cols = it.cycle(st.columns(3))
    for key, img in data.items():
        with next(cols):
            st.image(img, caption=f"{key} {img.shape}, {img.dtype}", use_container_width=True)


@st.cache_resource
def get_ds(img_dir: Path, shuffle: bool) -> tf.data.Dataset:
    tfds = sp.data.TFRecordDataset.get(img_dir, num_parallel_files=16)
    if shuffle:
        tfds = tfds.shuffle(100)
    return iter(tfds.prefetch(tf.data.AUTOTUNE))


@st.cache_data(max_entries=20)
def load_data(img_dir: Path, idx: np.int64, shuffle: bool) -> dict[str, np.ndarray]:
    # idx only keys the cache; the dataset is consumed sequentially
    tfds = get_ds(img_dir, shuffle)
    return next(tfds)


@st.cache_data(max_entries=20)
def get_data(
    img_dir: Path,
    idx: np.int64,
    use_bbox: bool = False,
    use_pose: bool = False,
    shuffle: bool = False,
) -> dict[str, np.ndarray]:

    data = load_data(img_dir, idx, shuffle=shuffle)

    if data is None:
        raise RuntimeError("Could not find data point.")

    sp_keys = sp.data.Dataset

    output = {}
    if sp_keys.RGB in data:
        output["RGB"] = data[sp_keys.RGB].numpy()

    if sp_keys.RGB_R in data:
        output["RGB_R"] = data[sp_keys.RGB_R].numpy()

    max_depth = 1.5

    def colorize_depth(arr: np.ndarray) -> np.ndarray:
        # v4 stores depth as float16, and cv2.convertScaleAbs has no float16 kernel
        # (it fails the "func != 0" assertion), so widen before handing it to OpenCV
        return cv2.applyColorMap(
            cv2.convertScaleAbs(arr.astype(np.float32), alpha=255 / max_depth),
            cv2.COLORMAP_JET,  # type: ignore
        )

    for key, label in (
        (sp_keys.DEPTH, "Depth"),
        (sp_keys.DEPTH_R, "Depth_R"),
        (sp_keys.DEPTH_GT, "Depth GT"),
        (sp_keys.DEPTH_GT_R, "Depth GT_R"),
    ):
        if key in data:
            output[label] = colorize_depth(data[key].numpy())

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

    # downscale here, inside the cache, so selectbox-only reruns don't re-resize
    return {k: resize_for_display(v) for k, v in output.items()}


def resize_for_display(img: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Downscale before sending to the browser; encoding/transferring full-res
    renders on every rerun is the main cost once the dataset pipeline is cached."""
    height, width = img.shape[:2]
    if width <= max_width:
        return img
    scale = max_width / width
    return cv2.resize(img, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def assemble_semantic_mask(mask, objs_data) -> np.ndarray:
    cls_colors = st.session_state["cls_colors"]

    # one obj_id -> colour lookup table, applied in a single pass; the naive
    # version did a full-frame `mask == obj_id` compare per object
    max_id = int(mask.max())
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for obj_data in objs_data:
        cls = obj_data["class"]
        cls_colors.setdefault(cls, np.random.randint(0, 256, size=3).astype(np.uint8).tolist())
        obj_id = int(obj_data["obj_id"])
        if 0 <= obj_id <= max_id:
            lut[obj_id] = cls_colors[cls]

    return lut[np.squeeze(mask)]


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
    if "index" not in st.session_state:
        st.session_state["index"] = 0
    main()
