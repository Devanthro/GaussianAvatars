# Avatar list for the operator app

The operator app (TeleportVR) used to hardcode the faces it offers. Instead it
now asks the robot which avatars are installed and gets their thumbnails with
the list, so a newly trained avatar shows up without an app release, and the app
cannot offer a face the robot does not have.

Everything uses stock ROS 2 types — ROS 2 has no standard string-array message,
and a custom one would need an interface package in the ROS-TCP-Endpoint
workspace plus generated C# classes in Unity. All three interfaces carry the
same JSON payload.

`{robot}` is the robot name, e.g. `charlie` (see [Robot name](#robot-name)).

| Interface | Type | Purpose |
| --- | --- | --- |
| `/robody/{robot}/avatar/list` | `std_srvs/srv/Trigger` | ask for the list |
| `/robody/{robot}/avatar/list/reply` | `std_msgs/msg/String` | the list, pushed on change |
| `/robody/{robot}/avatar/list/request` | `std_msgs/msg/Empty` | ask for the list over a topic |

## Service: `/robody/{robot}/avatar/list`

`success = true` and `message` = the JSON payload. If the node has not finished
its first scan of the avatar directory, `success = false` and `message` holds
the reason. The payload is built once per scan and cached, so the call returns
immediately.

```shell
ros2 service call /robody/charlie/avatar/list std_srvs/srv/Trigger
```

## Topic: `/robody/{robot}/avatar/list/reply`

`std_msgs/msg/String`, `data` = the JSON payload.

Published once when the node is ready, whenever the list or `active` changes,
and once per message on the request topic. **Never on a timer** — with nothing
changing and nobody asking, the topic stays silent.

QoS is reliable, transient local, depth 1, so a native ROS 2 subscriber that
joins late still receives the last list:

```shell
ros2 topic echo /robody/charlie/avatar/list/reply   # prints the last list at once
```

ROS-TCP-Endpoint subscribes as volatile and therefore does not get that last
sample, which is why the app also has the service and the request topic.

## Topic: `/robody/{robot}/avatar/list/request`

`std_msgs/msg/Empty`. Each message causes exactly one publish on
`/avatar/list/reply`:

```shell
ros2 topic pub --once /robody/charlie/avatar/list/request std_msgs/msg/Empty
```

## JSON payload

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

`active` follows the selection accepted on `/operator/name`: the node keeps that
selection while nobody is present and renders it as soon as presence arrives.
With no selection, the AI face is what is on screen and is reported as long as
the app can select it; otherwise `active` is `""`. The app keeps its own local
*None* entry, and the node keeps handling *None* on `/operator/name` as before.

## Where the list comes from

`avatar_registry.py` scans the avatar directory (`/workspace/avatars` by
default). A subdirectory holding a `.ply` is an avatar; anything else in the
directory is ignored, and an avatar that fails to load is left out and logged.

Optional files per avatar:

| File | Effect | Fallback |
| --- | --- | --- |
| `avatar.json` | `{"display_name": "Jürgen"}` | `display_name` = the id |
| `thumbnail.jpg` | any size, centre-cropped and scaled to 192×192 (`.jpeg`/`.png` work too) | a frontal frame of the neutral face, pre-rendered by `render_thumbnails.py`, else a generated placeholder tile |

## Rendering the thumbnails

The thumbnail is only ever displayed in the headset, but rendering it needs the
splat on a GPU, so `render_thumbnails.py` does it offline and writes
`thumbnail.jpg` next to the avatar:

```shell
python3 render_thumbnails.py                     # every avatar that has none
python3 render_thumbnails.py --avatar TA2R-57     # just this one
python3 render_thumbnails.py --force              # re-render everything
python3 render_thumbnails.py --dry-run            # only report what it would do
```

It needs CUDA and `diff_gaussian_rasterization`, so run it on the robot or on a
training machine. Each avatar is loaded into a throw-away model, posed neutral
(the same resting expression and tracked rest pose the viewer uses, see
`--no-resting-smile` and `--no-tracked-rest-pose`), rendered at 512² with the
camera distance and field of view from `run.sh`, then scaled to 192×192. The
CRT grid, vignette and white-key post-processing are deliberately skipped —
they are there for the robot's screen, not for a small tile. The running node
notices the new files within seconds and publishes a new `revision`; no restart.

Run it once after training an avatar and the robot never renders thumbnails at
runtime. That is the point: the viewer keeps a single `self.gaussians`, so
rendering avatar B's thumbnail inside the live process would swap out the face
currently on screen.

If you do want the node to render missing thumbnails itself, the hook is
already wired: give `LocalViewer` a method
`render_avatar_thumbnail(avatar_id, folder)` and `ros2_blendshape_node.py`
picks it up automatically (`getattr(viewer, 'render_avatar_thumbnail', None)`),
calling it once per avatar and caching the result in
`<avatar root>/.thumbnail_cache/<id>.jpg`. It should return a PIL image, an
HxWx3 uint8 numpy array or encoded image bytes of any size, or `None`.
`render_thumbnails.render_neutral_frame(folder)` is that render, ready to call.
Two caveats: it is called from the avatar-list scan thread, so it must not
touch dearpygui, and it needs enough free VRAM for a second model while one
avatar is already loaded — which is why pre-rendering is the better path.

**TODO(robot):** `render_thumbnails.py` has not been run on a GPU yet. It
follows the viewer's render path (`init_gaussians` → `reset_flame_param` →
`prepare_camera` → `gaussian_renderer.render`), but the framing of the result
wants one look before it is used for every avatar — adjust `--radius` and
`--fovy` if the head sits wrong in the tile.

### Ids

An avatar is addressed by its folder name, except for the faces the app has
been sending since before this list existed. Those keep their old ids through
`NAME_MAP` in `ros2_blendshape_node.py`:

| id | folder |
| --- | --- |
| `raf` | `rafael3` |
| `alona` | `alona5` |
| `caya` | `ai` (the AI face, rendered by the node, no folder needed) |
| `female` | `tanja_caritas_3` |
| `male` | `timo_2` |
| `johannes` | `timo` |
| `juri` | `kilian_3` |

The node accepts all of these plus the plain folder name, case-insensitively,
so app builds already in use keep working while a new avatar such as `TA2R-57`
is selectable right away. Legacy faces are listed first, in the order of
`NAME_MAP`; avatars found by folder name follow, sorted by display name.

### Rescanning

A background thread fingerprints the avatar directory every 2 s (names, mtimes,
and the mtime/size of each `avatar.json` and thumbnail) and rescans when that
fingerprint changes, plus a full rescan at least every 20 s. Adding or removing
an avatar folder therefore changes `revision` and publishes within seconds, with
no restart. Scanning runs off the ROS thread so it never stalls the render loop.

### Robot name

Resolved in this order: the `robot_name` ROS parameter, the `ROBOT_NAME`,
`ROBODY_ROBOT_NAME` or `ROBODY_NAME` environment variable, the hostname, then
`charlie`.

**TODO(robot):** read it from the same config `robody-hal` uses and put that
first in the chain (`resolve_robot_name()` in `ros2_avatar_list.py`) — the path
and format of that config are not known in this repository.

## Checking it without ROS

```shell
python3 avatar_registry.py --root /workspace/avatars        # summary
python3 avatar_registry.py --root /workspace/avatars --full # the payload itself
```

## Running it without the viewer

```shell
python3 ros2_avatar_list.py --robot charlie --root /workspace/avatars
```

Normally the list is part of the viewer's ROS node (`ros2_blendshape_node.py`),
so it starts and stops with the renderer.

The interfaces and the JSON schema are also summarised in the
[README](../README.md#avatar-list-for-the-operator-app).

## Files

| File | Contents |
| --- | --- |
| `avatar_registry.py` | directory scan, thumbnails, payload and `revision`; no ROS dependency, so it can be run and tested on a workstation |
| `ros2_avatar_list.py` | the service and the two topics, QoS, the scan thread, robot-name resolution |
| `ros2_blendshape_node.py` | creates the interface, resolves `/operator/name` through it and reports `active` |
| `render_thumbnails.py` | offline renderer for `thumbnail.jpg`; needs a GPU, not imported by the node |
