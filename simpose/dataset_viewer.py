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

from streamlit_image_comparison import image_comparison


@st.cache_data(show_spinner="Reading files...")
def get_idx(img_dir):
    indices = list(
        set(
            int(x.stem.split("_")[1])
            for x in Path(img_dir + "/rgb").glob("*.png")  # will be problematic with stereo
        )
    )
    assert len(indices) > 0, "No images found! Is it the correct directory?"
    indices.sort()
    return indices


@click.command()
@click.argument("data_dir", type=click.Path(exists=True, path_type=Path))
def main(data_dir: Path):
    st.set_page_config(layout="wide", page_title="Dataset Viewer")

    c1, c2, c3 = st.columns(3)
    img_dir = c1.text_input(
        "Image directory",
        str(data_dir.resolve()),
        label_visibility="collapsed",
        placeholder="Dataset directory",
    )
    if c2.button("↻"):
        st.cache_data.clear()

    indices = get_idx(img_dir)

    if len(indices) == 1:
        indices += [indices[0]]

    idx = c1.select_slider("Select image", indices, value=indices[0], key="idx")

    st.header(f"Datapoint: #{idx:05} (of total {len(indices)} images)")

    data = load_data(img_dir, idx)
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


def load_data(img_dir, idx):
    if "cls_colors" not in st.session_state:
        st.session_state["cls_colors"] = {
            "cpsduck": (0, 250, 250),
            "stapler": (162, 2, 20),
            "glue": (215, 235, 250),
            "chew_toy": (7, 7, 116),
            "wrench_13": (150, 150, 150),
            "pliers": (240, 255, 31),
            "lm_cam": (133, 133, 133),
            "lm_holepuncher": (242, 40, 13),
        }  # BGR

    cls_colors = st.session_state["cls_colors"]

    with open(os.path.join(img_dir, "gt", f"gt_{idx:05}.json")) as F:
        shot = json.load(F)
    cam_quat = shot["cam_rotation"]
    cam_matrix = np.array(shot["cam_matrix"])
    cam_rot = R.from_quat(cam_quat).as_matrix()
    cam_pos = np.array(shot["cam_location"])

    objs = shot["objs"]
    bgr = cv2.imread(os.path.join(img_dir, "rgb", f"rgb_{idx:04}.png"), cv2.IMREAD_ANYCOLOR)
    try:
        bgr_R = cv2.imread(
            os.path.join(img_dir, "rgb", f"rgb_{idx:04}_R.png"), cv2.IMREAD_ANYCOLOR
        )
    except Exception:
        bgr_R = None

    mask_path = Path(img_dir).joinpath(f"mask/mask_{idx:04}.exr")
    mask = EXR(mask_path).read("visib.R").astype(np.uint8)

    colored_mask_bgr = cv2.applyColorMap(
        cv2.convertScaleAbs(mask, alpha=255.0 / np.max(mask)),  # type: ignore
        cv2.COLORMAP_TURBO,
    )

    depth = np.array(
        cv2.imread(
            os.path.join(img_dir, "depth", f"depth_{idx:04}.exr"),
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )
    ).astype(np.float32)

    depth[depth > 50.0] = 0.0
    colored_depth = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=255 / np.max(depth)), cv2.COLORMAP_JET  # type: ignore
    )

    colored_semantic_mask_bgr = np.zeros((*mask.shape[:2], 3)).astype(np.uint8)

    assert bgr is not None, f"Could not load image for {id:04}"

    for obj in objs:
        # semantics
        cls = obj["class"]
        cls_colors.setdefault(cls, np.random.randint(0, 255, size=3).astype(np.uint8).tolist())
        colored_semantic_mask_bgr[mask == obj["object id"]] = cls_colors[cls]

        # bbox
        bbox = obj["bbox_visib"]
        # bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        # "visib_fract": visib_fract,
        if obj["visib_fract"] > 0.1:
            cv2.rectangle(
                bgr,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color=cls_colors[cls],
                thickness=2,
            )

        obj_pos = np.array(obj["pos"])
        quat = obj["rotation"]
        obj_rot = R.from_quat(quat).as_matrix()  # w, x, y, z -> x, y, z, w
        t = cam_rot.T @ (obj_pos - cam_pos)
        RotM = cam_rot.T @ obj_rot

        rotV, _ = cv2.Rodrigues(RotM)
        cv2.drawFrameAxes(
            bgr, cameraMatrix=cam_matrix, rvec=rotV, tvec=t, distCoeffs=0, length=0.05, thickness=1
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


if __name__ == "__main__":
    main()
