import json
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import click
from pathlib import Path


@click.command()
@click.argument("img_dir", type=click.Path(exists=True))
def main(img_dir):
    idxs = [
        int(x.stem.split("_")[1])
        for x in Path(img_dir+"/rgb").glob("*.png") # will be problematic with stereo
    ]

    cls_ids = {
        "cpsduck": 1,
        "stapler": 2,
        "glue": 3,
        "chew_toy": 4,
        "wrench_13": 5,
        "pliers": 6,
    }
    cls_colors = {
        1: (0, 250, 250),
        2: (162, 2, 20),
        3: (215, 235, 250),
        4: (7, 7, 116),
        5: (150, 150, 150),
        6: (240, 255, 31),
    }  # BGR

    try:
        for idx in idxs:
            with open(os.path.join(img_dir, "gt", f"gt_{idx:05}.json")) as F:
                shot = json.load(F)
            cam_quat = shot["cam_rotation"]
            cam_matrix = np.array(shot["cam_matrix"])
            cam_rot = R.from_quat(cam_quat).as_matrix()
            cam_pos = np.array(shot["cam_location"])

            objs = shot["objs"]
            bgr = cv2.imread(
                os.path.join(img_dir, "rgb", f"rgb_{idx:04}.png"), cv2.IMREAD_ANYCOLOR
            )

            mask = cv2.imread(
                os.path.join(img_dir, "mask", f"mask_{idx:04}.exr"),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            )

            colored_mask_bgr = cv2.applyColorMap(
                cv2.convertScaleAbs(mask, alpha=255 / np.max(mask)),
                cv2.COLORMAP_RAINBOW,
            )

            depth = cv2.imread(
                os.path.join(img_dir, "depth", f"depth_{idx:04}.exr"),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            )
            depth[depth > 50.0] = 0.0
            colored_depth = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=255 / np.max(depth)), cv2.COLORMAP_JET
            )

            semantic_mask = mask.copy()

            if bgr is None:
                print(f"Could not load image for {id:04}")
                continue

            assert bgr is not None, f"Could not load image for {id:04}"

            for obj in objs:
                # semantics
                cls = cls_ids[obj["class"]]
                semantic_mask[mask == obj["object id"]] = cls

                # bbox
                bbox = obj["bbox_visib"]
                # bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                # "visib_fract": visib_fract,
                if obj["visib_fract"] > 0.5:
                    cv2.rectangle(
                        bgr,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        color=cls_colors[cls],
                        thickness = 2
                    )

                obj_pos = np.array(obj["pos"])
                quat = obj["rotation"]
                obj_rot = R.from_quat(quat).as_matrix()  # w, x, y, z -> x, y, z, w
                t = cam_rot.T @ (obj_pos - cam_pos)
                RotM = cam_rot.T @ obj_rot

                # rotV, _ = cv2.Rodrigues(RotM)
                # cv2.drawFrameAxes(
                #     rgb,
                #     cameraMatrix=cam_matrix,
                #     rvec=rotV,
                #     tvec=t,
                #     distCoeffs=0,
                #     length=0.1,
                # )

            colored_semantic_mask_bgr = np.zeros_like(bgr)
            for cls in cls_ids.values():
                colored_semantic_mask_bgr[semantic_mask[..., 0] == cls] = cls_colors[
                    cls
                ]

            print(f"Showing image: {idx:04}")

            # create preview, with rgb and mask
            row1 = np.hstack((bgr, colored_mask_bgr))
            row2 = np.hstack((colored_semantic_mask_bgr, colored_depth))
            preview = np.vstack((row1, row2))

            cv2.imshow(f"Preview", preview)

            key = cv2.waitKey(0)

            if key == ord("q") or key == 27:  # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
