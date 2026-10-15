#!/bin/bash
source /opt/ros/humble/install/setup.bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <avatar_name>"
    exit 1
fi

AVATAR_NAME=$1

python local_viewer.py --point_path /workspace/avatars/$AVATAR_NAME/point_cloud.ply --sh-degree 0 --radius 0.65 --background-color 0 0 0 --demo-mode --tracked-rest-pose \
    --grid-spacing 6 --grid-alpha 0.75 --grid-sharpness 0.8 \
    --vignette-strength 0.92 --vignette-radius 0.57 --vignette-end 1.1 --vignette-shape 2 \
    --white-key-chroma 0.15 --white-key-radius 0.55
