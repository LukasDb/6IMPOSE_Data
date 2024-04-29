import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import streamlit as st
import click
import time
import simpose as sp

from streamlit_image_comparison import image_comparison
import tensorflow as tf


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

    data = load_data(Path(img_dir), idx, use_bbox=use_bbox, use_pose=use_pose)

    rgb = data["rgb"]
    rgb_R = data["rgb_R"]
    depth = data["depth"]
    colored_depth = data["colored_depth"]
    colored_mask_rgb = data["colored_mask_rgb"]
    colored_semantic_mask_rgb = data["colored_semantic_mask_rgb"]
    mask = data["mask"]

    img2name = {
        "rgb": "RGB",
        "rgb_R": "RGB_R",
        "colored_depth": "Depth",
        "colored_mask_rgb": "Instance",
        "colored_semantic_mask_rgb": "Semantic",
    }
    if data["rgb_R"] is None:
        img2name.pop("rgb_R")

    chose_col1, chose_col2 = c2.columns(2)
    chosen_left = chose_col1.selectbox(
        "Select view",
        options=list(img2name.keys()),
        index=0,
        format_func=lambda x: img2name[x],
        key="view_left",
    )
    chosen_right = chose_col2.selectbox(
        "Select view",
        options=list(img2name.keys()),
        index=1,
        format_func=lambda x: img2name[x],
        key="view_right",
    )
    if chosen_left is None:
        chosen_left = "rgb"

    if chosen_right is None:
        chosen_right = "colored_depth"

    left_img = data[chosen_left]
    right_img = data[chosen_right]

    # create preview, with rgb and mask
    image_comparison(
        img1=left_img,
        img2=right_img,
        label1=chosen_left,
        label2=chosen_right,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.image(rgb, caption=f"RGB {rgb.shape}, {rgb.dtype}", use_column_width=True)
        if rgb_R is not None:
            st.image(rgb_R, caption=f"RGB_R {rgb_R.shape}, {rgb_R.dtype}", use_column_width=True)
        st.image(
            colored_depth, caption=f"Depth {depth.shape}, {depth.dtype}", use_column_width=True
        )

    with c2:
        if mask is not None:
            st.image(
                colored_mask_rgb,
                caption=f"Instance Mask {mask.shape}, {mask.dtype}",
                use_column_width=True,
            )
            st.image(
                colored_semantic_mask_rgb,
                caption=f"Semantic Mask {mask.shape}, {mask.dtype}",
                use_column_width=True,
            )


def create_visualization(
    bgr: np.ndarray,
    bgr_R: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray | None,
    cam_data: dict[str, np.ndarray],
    objs_data: list[dict] | None,
    use_bbox: bool,
    use_pose: bool,
) -> dict[str, np.ndarray]:
    cam_matrix = cam_data["cam_matrix"]
    cam_rot = R.from_quat(cam_data["cam_rot"]).as_matrix()
    cam_pos = cam_data["cam_pos"]

    if "cls_colors" not in st.session_state:
        st.session_state["cls_colors"] = {}
    cls_colors = st.session_state["cls_colors"]

    if mask is not None:
        mask_scaled = cv2.convertScaleAbs(mask, alpha=255.0 / np.max(mask))
        colored_mask_bgr = cv2.applyColorMap(
            mask_scaled,
            cv2.COLORMAP_TURBO,
        )

    colored_depth = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=255 / np.max(depth)), cv2.COLORMAP_JET  # type: ignore
    )

    colored_semantic_mask_bgr = np.zeros((*bgr.shape[:2], 3)).astype(np.uint8)

    objs_data = [] if objs_data is None else objs_data

    for obj_data in objs_data:
        # semantic
        cls = obj_data["class"]
        cls_colors.setdefault(cls, np.random.randint(0, 256, size=3).astype(np.uint8).tolist())
        colored_semantic_mask_bgr[mask == obj_data["obj_id"]] = cls_colors[cls]

        # bbox
        if use_bbox:
            bbox = obj_data["bbox_visib"]
            bgr = cv2.rectangle(
                bgr.copy(),
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color=cls_colors[cls],
                thickness=2,
            )
            # write class name
            font = cv2.FONT_HERSHEY_SIMPLEX
            bgr = cv2.putText(
                bgr,
                str(cls),
                (bbox[0], bbox[1]),
                font,
                1,
                cls_colors[cls],
                2,
                cv2.LINE_AA,
            )

        if use_pose:
            obj_pos = np.array(obj_data["pos"])
            quat = obj_data["rotation"]
            obj_rot = R.from_quat(quat).as_matrix()  # w, x, y, z -> x, y, z, w
            t = cam_rot.T @ (obj_pos - cam_pos)
            RotM = cam_rot.T @ obj_rot

            rotV, _ = cv2.Rodrigues(RotM)
            bgr = cv2.drawFrameAxes(
                bgr.copy(),
                cameraMatrix=cam_matrix,
                rvec=rotV,
                tvec=t,
                distCoeffs=np.zeros(5),
                length=0.05,
                thickness=2,
            )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_R = cv2.cvtColor(bgr_R, cv2.COLOR_BGR2RGB) if bgr_R is not None else None
    if mask is not None:
        colored_mask_rgb = cv2.cvtColor(colored_mask_bgr, cv2.COLOR_BGR2RGB)
        colored_semantic_mask_rgb = cv2.cvtColor(colored_semantic_mask_bgr, cv2.COLOR_BGR2RGB)
    else:
        colored_mask_rgb = None
        colored_semantic_mask_rgb = None

    return {
        "rgb": rgb,
        "rgb_R": rgb_R,
        "mask": mask,
        "colored_mask_rgb": colored_mask_rgb,
        "colored_semantic_mask_rgb": colored_semantic_mask_rgb,
        "depth": depth,
        "colored_depth": colored_depth,
    }


@st.cache_data()
def load_data(
    img_dir: Path, idx: np.int64, use_bbox: bool = False, use_pose: bool = False
) -> dict[str, np.ndarray]:

    tfds = sp.data.TFRecordDataset.get(img_dir, num_parallel_files=1)
    data = tfds.skip(idx).take(1).get_single_element()

    if data is None:
        raise RuntimeError("Could not find data point.")

    cam_data = {
        "cam_matrix": data["cam_matrix"].numpy(),
        "cam_rot": data["cam_rotation"].numpy(),
        "cam_pos": data["cam_location"].numpy(),
    }

    bgr = cv2.cvtColor(data["rgb"].numpy(), cv2.COLOR_RGB2BGR)
    bgr_R = cv2.cvtColor(data["rgb_R"].numpy(), cv2.COLOR_RGB2BGR) if "rgb_R" in data else None
    depth = data["depth"].numpy()
    mask = None if "mask" not in data else data["mask"].numpy()

    if (
        "obj_classes" in data
        and "obj_ids" in data
        and "obj_pos" in data
        and "obj_rot" in data
        and "obj_bbox_visib" in data
        and "obj_visib_fract" in data
    ):
        objs_data = [
            {
                "class": cls,
                "obj_id": obj_id,
                "pos": pos,
                "rotation": rot,
                "bbox_visib": bbox_visib,
                "visib_fract": visib_fract,
            }
            for cls, obj_id, pos, rot, bbox_visib, visib_fract in zip(
                data["obj_classes"].numpy(),
                data["obj_ids"].numpy(),
                data["obj_pos"].numpy(),
                data["obj_rot"].numpy(),
                data["obj_bbox_visib"].numpy(),
                data["obj_visib_fract"].numpy(),
            )
        ]
    else:
        objs_data = []

    return create_visualization(
        bgr, bgr_R, depth, mask, cam_data, objs_data, use_bbox=use_bbox, use_pose=use_pose
    )


if __name__ == "__main__":
    main()
