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
@click.option("--mp4", is_flag=True, help="Generate a mp4 video of the dataset instead")
def main(img_dir, mp4):
    idxs = [
        int(x.stem.split("_")[1])
        for x in Path(img_dir + "/rgb").glob("*.png")  # will be problematic with stereo
    ]
    idxs.sort()

    cls_colors = {
        "cpsduck": (0, 250, 250),
        "stapler": (162, 2, 20),
        "glue": (215, 235, 250),
        "chew_toy": (7, 7, 116),
        "wrench_13": (150, 150, 150),
        "pliers": (240, 255, 31),
        "lm_cam": (133, 133, 133),
    }  # BGR

    if mp4:
        images = list(Path(img_dir + "/rgb").glob("*.png"))
        images.sort()

        frame = cv2.imread(str(images[0]))
        height, width, layers = frame.shape

        FPS = 24
        video = cv2.VideoWriter(
            "out.mp4",
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            FPS,
            (width * 2, height * 2),
        )

    try:
        idx = 0
        while True:
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

            colored_semantic_mask_bgr = np.zeros_like(bgr)
            
            if bgr is None:
                print(f"Could not load image for {id:04}")
                continue

            assert bgr is not None, f"Could not load image for {id:04}"

            for obj in objs:
                # semantics
                cls = obj["class"]                
                colored_semantic_mask_bgr[mask[..., 0] == obj["object id"]] = cls_colors[cls]

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

                # rotV, _ = cv2.Rodrigues(RotM)
                # cv2.drawFrameAxes(
                #     bgr,
                #     cameraMatrix=cam_matrix,
                #     rvec=rotV,
                #     tvec=t,
                #     distCoeffs=0,
                #     length=0.1,
                # )

            # create preview, with rgb and mask
            row1 = np.hstack((bgr, colored_mask_bgr))
            row2 = np.hstack((colored_semantic_mask_bgr, colored_depth))
            preview = np.vstack((row1, row2))

            print("\r" + f"Image: {idx:05}/{len(idxs):05}", end="")

            if not mp4:
                cv2.imshow(f"Preview", preview)
                key = cv2.waitKey(0)
                if key == ord("q") or key == 27:  # ESC
                    break

                elif key == ord("a"):
                    idx -= 2
            else:
                # cv2.waitKey(1)
                video.write(preview)

            idx += 1

            idx = np.clip(idx, 0, len(idxs) - 1)

    except KeyboardInterrupt:
        pass
    finally:
        # bar.close()
        cv2.destroyAllWindows()
        if mp4:
            video.release()


if __name__ == "__main__":
    main()
