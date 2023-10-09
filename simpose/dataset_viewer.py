import json
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import streamlit as st
import click
from simpose.exr import EXR
import h5py
import time

from streamlit_image_comparison import image_comparison


@st.cache_data()
def get_idx(img_dir):
    if (img_dir / "data.h5").exists():
        F = None
        with st.spinner("Waiting for free h5 file..."):
            while F is None:
                try:
                    F = h5py.File(img_dir / "data.h5", "r")
                except BlockingIOError:
                    time.sleep(0.01)

        existing_ids = F["indices"]
        assert isinstance(existing_ids, h5py.Dataset)
        existing_ids = np.unique(existing_ids)
        F.close()
        return existing_ids
    
    # fall back to simpose
    indices = list(
        set(
            int(x.stem.split("_")[1])
            for x in Path(img_dir / "rgb").glob("*.png")  # will be problematic with stereo
        )
    )
    assert len(indices) > 0, "No images found! Is it the correct directory?"
    indices.sort()
    return indices


@click.command()
@click.argument("data_dir", type=click.Path(exists=True, path_type=Path))
def main(data_dir: Path):
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
        st.image(rgb, caption=f"RGB {rgb.shape}, {rgb.dtype}")
        if rgb_R is not None:
            st.image(rgb_R, caption=f"RGB_R {rgb_R.shape}, {rgb_R.dtype}")
        st.image(colored_depth, caption=f"Depth {depth.shape}, {depth.dtype}")

    with c2:
        st.image(colored_mask_rgb, caption=f"Instance Mask {mask.shape}, {mask.dtype}")
        st.image(colored_semantic_mask_rgb, caption=f"Semantic Mask {mask.shape}, {mask.dtype}")


def create_visualization(bgr, bgr_R, depth, mask, cam_data, objs_data, use_bbox, use_pose):
    cam_matrix = cam_data["cam_matrix"]
    cam_rot = R.from_quat(cam_data["cam_rot"]).as_matrix()
    cam_pos = cam_data["cam_pos"]

    if "cls_colors" not in st.session_state:
        st.session_state["cls_colors"] = {}
    cls_colors = st.session_state["cls_colors"]

    mask_scaled = cv2.convertScaleAbs(mask, alpha=255.0 / np.max(mask))
    colored_mask_bgr = cv2.applyColorMap(
        mask_scaled,
        cv2.COLORMAP_TURBO,
    )
    colored_depth = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=255 / np.max(depth)), cv2.COLORMAP_JET  # type: ignore
    )

    colored_semantic_mask_bgr = np.zeros((*mask.shape[:2], 3)).astype(np.uint8)

    for obj_data in objs_data:
        # semantic
        cls = obj_data["class"]
        cls_colors.setdefault(cls, np.random.randint(0, 256, size=3).astype(np.uint8).tolist())
        colored_semantic_mask_bgr[mask == obj_data["object id"]] = cls_colors[cls]

        # bbox
        if use_bbox:
            bbox = obj_data["bbox_visib"]
            # bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # "visib_fract": visib_fract,
            if obj_data["visib_fract"] > 0.1:
                bgr = cv2.rectangle(
                    bgr.copy(),
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    color=cls_colors[cls],
                    thickness=2,
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
    colored_mask_rgb = cv2.cvtColor(colored_mask_bgr, cv2.COLOR_BGR2RGB)
    colored_semantic_mask_rgb = cv2.cvtColor(colored_semantic_mask_bgr, cv2.COLOR_BGR2RGB)

    return {
        "rgb": rgb,
        "rgb_R": rgb_R,
        "mask": mask,
        "colored_mask_rgb": colored_mask_rgb,
        "colored_semantic_mask_rgb": colored_semantic_mask_rgb,
        "depth": depth,
        "colored_depth": colored_depth,
    }


def load_data(img_dir: Path, idx: int, use_bbox=False, use_pose=False):
    if st.session_state["last_idx"] != idx or "loaded_data" not in st.session_state:
        st.session_state["last_idx"] = idx

        if (Path(img_dir) / "data.h5").exists():
            loaded_data = load_data_h5(img_dir, idx)
        else:
            loaded_data = load_data_simpose(img_dir, idx)

        st.session_state["loaded_data"] = loaded_data

    loaded_data = st.session_state["loaded_data"]
    return create_visualization(*loaded_data, use_bbox, use_pose)


def load_data_simpose(img_dir, idx):
    with open(os.path.join(img_dir, "gt", f"gt_{idx:05}.json")) as F:
        shot = json.load(F)
    cam_quat = shot["cam_rotation"]
    cam_data = {
        "cam_matrix": np.array(shot["cam_matrix"]),
        "cam_rot": cam_quat,
        "cam_pos": np.array(shot["cam_location"]),
    }

    bgr = cv2.imread(os.path.join(img_dir, "rgb", f"rgb_{idx:04}.png"), cv2.IMREAD_ANYCOLOR)
    try:
        bgr_R = cv2.imread(
            os.path.join(img_dir, "rgb", f"rgb_{idx:04}_R.png"), cv2.IMREAD_ANYCOLOR
        )
    except Exception:
        bgr_R = None

    mask_path = Path(img_dir).joinpath(f"mask/mask_{idx:04}.exr")
    mask = EXR(mask_path).read("visib.R").astype(np.uint8)

    depth = np.array(
        cv2.imread(
            os.path.join(img_dir, "depth", f"depth_{idx:04}.exr"),
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )
    ).astype(np.float32)

    objs = shot["objs"]

    objs_data = [
        {
            "class": d["class"],
            "object id": d["object id"],
            "pos": d["pos"],
            "rotation": d["rotation"],
            "bbox_visib": d["bbox_visib"],
            "visib_fract": d["visib_fract"],
        }
        for d in objs
    ]

    return bgr, bgr_R, depth, mask, cam_data, objs_data


def load_data_h5(img_dir, idx):
    F = None
    with st.spinner("Waiting for free h5 file..."):
        while F is None:
            try:
                F = h5py.File(img_dir / "data.h5", "r")
            except BlockingIOError:
                time.sleep(0.01)

    cam_matrix_ds = F["cam_matrix"]
    cam_pos_ds = F["cam_pos"]
    cam_rot_ds = F["cam_rot"]
    assert isinstance(cam_matrix_ds, h5py.Dataset)
    assert isinstance(cam_pos_ds, h5py.Dataset)
    assert isinstance(cam_rot_ds, h5py.Dataset)

    cam_data = {
        "cam_matrix": cam_matrix_ds[idx],
        "cam_rot": cam_rot_ds[idx],
        "cam_pos": cam_pos_ds[idx],
    }

    rgb_ds = F["rgb"]
    assert isinstance(rgb_ds, h5py.Dataset)
    bgr = cv2.cvtColor(rgb_ds[idx], cv2.COLOR_RGB2BGR)
    if "rgb_R" in F.keys():
        rgb_R_ds = F["rgb_R"]
        assert isinstance(rgb_R_ds, h5py.Dataset)
        bgr_R = cv2.cvtColor(rgb_R_ds[idx], cv2.COLOR_RGB2BGR)
    else:
        bgr_R = None

    mask_ds = F["mask"]
    assert isinstance(mask_ds, h5py.Dataset)
    mask = mask_ds[idx].astype(np.float32)

    depth_ds = F["depth"]
    assert isinstance(depth_ds, h5py.Dataset)
    depth = depth_ds[idx]

    objs_group = F["objs"]
    assert isinstance(objs_group, h5py.Group)
    objs = objs_group[f"{idx:06}"]
    assert isinstance(objs, h5py.Group)

    cls_ds = objs["class"]
    id_ds = objs["object id"]
    pos_ds = objs["pos"]
    rot_ds = objs["rotation"]
    bbox_ds = objs["bbox_visib"]
    visib_ds = objs["visib_fract"]

    assert isinstance(cls_ds, h5py.Dataset)
    assert isinstance(id_ds, h5py.Dataset)
    assert isinstance(pos_ds, h5py.Dataset)
    assert isinstance(rot_ds, h5py.Dataset)
    assert isinstance(bbox_ds, h5py.Dataset)
    assert isinstance(visib_ds, h5py.Dataset)

    objs_data = [
        {
            "class": cls_ds[i],
            "object id": id_ds[i],
            "pos": pos_ds[i],
            "rotation": rot_ds[i],
            "bbox_visib": bbox_ds[i],
            "visib_fract": visib_ds[i],
        }
        for i in range(len(cls_ds))
    ]

    F.close()

    return bgr, bgr_R, depth, mask, cam_data, objs_data


if __name__ == "__main__":
    main()
