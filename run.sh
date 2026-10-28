#!/bin/bash
source /opt/ros/humble/install/setup.bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <avatar_name>"
    exit 1
fi

AVATAR_NAME=$1

# The robot name in the avatar list topics, /robody/<robot>/avatar/list, comes
# from robody-hal's /etc/robody-ta2/hal.env (see ros2_avatar_list.py).

# Thumbnails for the operator app's avatar picker: render the missing ones, keep
# every existing one. Needs the avatar directory writable; a failure here must
# not keep the face from starting.
if [ -w /workspace/avatars ]; then
    python3 render_thumbnails.py || echo "render_thumbnails.py failed; starting the viewer anyway" >&2
else
    echo "/workspace/avatars is read-only; not rendering missing thumbnails" >&2
fi

python local_viewer.py --point_path /workspace/avatars/$AVATAR_NAME/point_cloud.ply --sh-degree 0 --radius 0.65 --background-color 0 0 0 --demo-mode --tracked-rest-pose \
    --grid-spacing 6 --grid-alpha 0.75 --grid-sharpness 0.8 \
    --vignette-strength 0.92 --vignette-radius 0.57 --vignette-end 1.1 --vignette-shape 2 \
    --white-key-chroma 0.15 --white-key-radius 0.55
