# GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians

<div align="center"> 
  <img src="media/demo.gif">

  <br>

  [project](https://shenhanqian.github.io/gaussian-avatars) / [arxiv](http://arxiv.org/abs/2312.02069) / [video](https://www.youtube.com/watch?v=lVEY78RwU_I) / [face tracker](https://github.com/ShenhanQian/VHAP) / [bibtex](https://shenhanqian.github.io/raw.html?filePath=/assets/2023-12-04-gaussian-avatars/bibtex.bib)
</div>

## Licenses

This work is made available under [CC-BY-NC-SA-4.0](./LICENSE.md) and is subject to the following statement:

> Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to this software and related documentation. Any commercial use, reproduction, disclosure or distribution of this software and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.

This project uses [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), which carries its [original license](./LICENSE_GS.md).
The GUI is inspired by [INSTA](https://github.com/Zielon/INSTA). 
The mesh rendering operations are adapted from [NVDiffRec](https://github.com/NVlabs/nvdiffrec) and [NVDiffRast](https://github.com/NVlabs/nvdiffrast). 

![Method](media/method.jpg)

## Setup

### [1. Installation](doc/installation.md)

### [2. Download](doc/download.md)

## Usage

### 0. Demo
You can play with a trained GaussianAvatar without downloading the dataset:
```shell
python local_viewer.py --point_path media/306/point_cloud.ply
```

### 1. Training

```shell
SUBJECT=306

python train.py \
-s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
--eval --bind_to_mesh --white_background --port 60000
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments</span></summary>

- `--source_path` / `-s`

    Path to the source directory containing a COLMAP or Synthetic NeRF data set.

- `--model_path` / `-m`

    Path where the trained model should be stored (```output/<random>``` by default).

- `--eval`

   Add this flag to use a training/val/test split for evaluation. Otherwise, all images are used for training.

- `--bind_to_mesh`

  Add this flag to bind 3D Gaussians to a driving mesh, e.g., FLAME.

- `--resolution` / `-r`

  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**

- `--white_background` / `-w`

  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.

- `--sh_degree`

    Order of spherical harmonics to be used (no larger than 3). ```3``` by default.

- `--iterations`

  Number of total iterations to train for, ```30_000``` by default.

- `--port`

  Port to use for GUI server, ```60000``` by default.

</details>

> [!NOTE]
> During training, a complete evaluation are conducted on both the validation set (novel-view synthesis) and test set (self-reenactment) every `--interval` iterations. You can check the metrics in the commandline or Tensorboard. The metrics are computed on all images, although we only save partial images in Tensorboard.

### 2. Interactive Viewers

#### Remote Viewer

![remote viewer](media/remote_viewer.png)

During training, one can monitor the training progress with the remote viewer

```shell
python remote_viewer.py --port 60000
```

> [!NOTE]
> - The remote viewer can slow down training a lot. You may want to close it or check "pause rendering" when not viewing.
>
> - The viewer could get frozen and disconnected the first time you enable "show mesh". You can try switching it on and off or simply wait for a few seconds.

#### Local Viewer

![local viewer](media/local_viewer.png)

After training, one can load and render the optimized 3D Gaussians with the local viewer

```shell
SUBJECT=306
ITER=300000

python local_viewer.py \
--point_path output/UNION10EMOEXP_${SUBJECT}_eval_600k/point_cloud/iteration_${ITER}/point_cloud.ply
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments</span></summary>

- `--point_path`

  Path to the gaussian splatting file (ply)

- `--motion_path`

  Path to the motion file (npz). You only need this if you want to load a different motion sequence than the original one for training.

</details>


### [3. Offline Rendering](doc/offline_render.md)

## On the robot

### Avatar list for the operator app

The operator app (TeleportVR) used to hardcode the faces it offers. Instead it
asks the robot which avatars are installed and gets their thumbnails with the
list, so a newly trained avatar shows up without an app release, and the app
cannot offer a face the robot does not have.

Everything uses stock ROS 2 types — ROS 2 has no standard string-array message,
and a custom one would need an interface package in the ROS-TCP-Endpoint
workspace plus generated C# classes in Unity. All three interfaces carry the
same JSON payload. `{robot}` is the robot name, e.g. `charlie`.

| Interface | Type | Purpose |
| --- | --- | --- |
| `/robody/{robot}/avatar/list` | `std_srvs/srv/Trigger` | ask for the list |
| `/robody/{robot}/avatar/list/reply` | `std_msgs/msg/String` | the list, pushed on change |
| `/robody/{robot}/avatar/list/request` | `std_msgs/msg/Empty` | ask for the list over a topic |

```shell
# the list, as a service call
ros2 service call /robody/charlie/avatar/list std_srvs/srv/Trigger

# the list, over topics: one request gives exactly one reply
ros2 topic echo /robody/charlie/avatar/list/reply
ros2 topic pub --once /robody/charlie/avatar/list/request std_msgs/msg/Empty
```

The service answers `success = true` with the payload in `message`, or
`success = false` with the reason while the first directory scan is still
running.

`/avatar/list/reply` is published once when the node is ready, whenever the
list or `active` changes, and once per message on the request topic — **never
on a timer**. Its QoS is reliable, transient local, depth 1, so a native ROS 2
subscriber that joins late still receives the last list. ROS-TCP-Endpoint
subscribes as volatile and does not get that last sample, which is why the app
also has the service and the request topic.

#### JSON payload

```json
{
  "version": 1,
  "revision": "c41f9a",
  "active": "raf",
  "avatars": [
    {
      "id": "raf",
      "display_name": "Rafael",
      "thumbnail_jpeg_base64": "/9j/4AAQSkZJRgABAQ..."
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `version` | payload schema version, currently `1` |
| `revision` | hash over the ids, display names and thumbnails — **not** over `active`. The app re-decodes thumbnails only when it changes. |
| `active` | the avatar the robot is showing, or `""` |
| `avatars` | the list, in the order the app should show it |
| `avatars[].id` | exactly the string the node accepts on `/operator/name` and in the `name` field of `/operator/face/position` |
| `avatars[].display_name` | UTF-8 label shown in the headset, e.g. `"Jürgen"` |
| `avatars[].thumbnail_jpeg_base64` | square JPEG, 192×192, quality ~85, ≤ 20 KB before base64 |

The list comes from a scan of `/workspace/avatars`: a subdirectory holding a
`.ply` is an avatar, with optional `avatar.json` (`{"display_name": "Jürgen"}`)
and `thumbnail.jpg` next to it. The directory is rescanned on change and at
least every 20 s, so adding or removing an avatar changes `revision` and
publishes within seconds, with no restart.

Thumbnails are pre-rendered once per avatar, on a machine with a GPU, instead of
by the running node — an avatar without one falls back to a placeholder tile:

```shell
python3 render_thumbnails.py --avatar TA2R-57   # writes thumbnail.jpg
```

Avatars are addressed by folder name, except for the faces the app has been
sending since before this list existed (`raf`, `alona`, `caya`, `female`,
`male`, `johannes`, `juri`), which keep their ids — so app builds already in
use keep working.

**[Full documentation: doc/avatar_list.md](doc/avatar_list.md)** — ids and
folders, thumbnail fallbacks, rescan details, robot-name resolution and how to
inspect the payload without ROS.

## Cite

If you find our paper or code useful in your research, please cite with the following BibTeX entry:
```bibtex
@inproceedings{qian2024gaussianavatars,
  title={Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians},
  author={Qian, Shenhan and Kirschstein, Tobias and Schoneveld, Liam and Davoli, Davide and Giebenhain, Simon and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20299--20309},
  year={2024}
}
```
