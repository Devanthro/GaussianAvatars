#!/usr/bin/env python3
"""Pre-render the avatar thumbnails the operator app shows in its picker.

The thumbnail itself is only ever displayed in the headset, but rendering it
needs the splat on a GPU.  Doing that inside the running node is awkward: the
viewer keeps a single ``self.gaussians``, so rendering another avatar's
thumbnail would swap out the face currently on screen.  So render them once,
offline, and leave the result next to the avatar as ``thumbnail.jpg`` -- the
file ``avatar_registry`` prefers over anything it can generate itself.

Run it after training a new avatar, or once over the whole directory::

    python3 render_thumbnails.py                        # all avatars missing one
    python3 render_thumbnails.py --avatar TA2R-57       # just this one
    python3 render_thumbnails.py --force                # re-render everything

Needs CUDA and the ``diff_gaussian_rasterization`` extension, so run it on the
robot or on any training machine -- not on a laptop without a GPU.

The framing follows what the robot shows (``run.sh``): neutral face, the same
camera distance and field of view, black background.  The CRT grid, vignette
and white-key post-processing the viewer applies are deliberately skipped --
they are there for the robot's screen, not for a 192x192 tile.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from avatar_registry import (DEFAULT_AVATAR_ROOT, THUMBNAIL_FILENAMES, check_avatar_folder,
                             encode_thumbnail)

#: Rendered at this size, then centre-cropped and scaled down to 192x192. Well
#: above the final size so the downscale does the antialiasing.
DEFAULT_RENDER_SIZE = 512
#: Camera distance and field of view as run.sh uses them on the robot.
DEFAULT_RADIUS = 0.65
DEFAULT_FOVY = 20.0
DEFAULT_BACKGROUND = (0.0, 0.0, 0.0)
DEFAULT_SH_DEGREE = 0
#: ARKit blendshape count expected by convert_blendshapes_to_expressions().
NUM_ARKIT_BLENDSHAPES = 51


class PipelineConfig:
    """What ``gaussian_renderer.render`` reads off the pipeline object."""
    debug = False
    compute_cov3D_python = False
    convert_SHs_python = False


def neutral_flame_param(gaussians, tracked_rest_pose: bool = True,
                        resting_smile: bool = True) -> dict:
    """FLAME parameters for a neutral, forward-facing head.

    Mirrors ``LocalViewer.reset_flame_param`` and the zero-blendshape case of
    ``LocalViewer.update_flame_model`` so the thumbnail shows the same face the
    robot does when the operator is quiet.
    """
    import torch
    from scipy.spatial.transform import Rotation as R

    flame_param = {
        'expr': torch.zeros(1, gaussians.n_expr),
        'rotation': torch.zeros(1, 3),
        'neck': torch.zeros(1, 3),
        'jaw': torch.zeros(1, 3),
        'eyes': torch.zeros(1, 6),
        'translation': torch.zeros(1, 3),
    }

    # Zero blendshapes still map to a slightly expressive face; the viewer adds
    # the same offsets to avoid a dead stare ("resting bitchface").
    expressions, jaw = gaussians.flame_model.mask.convert_blendshapes_to_expressions(
        np.zeros(NUM_ARKIT_BLENDSHAPES, dtype=np.float32))
    flame_param['expr'] = expressions.unsqueeze(0)
    flame_param['jaw'][0, :len(jaw)] = jaw
    if resting_smile:
        flame_param['expr'][0, 0] += 1.5
        flame_param['expr'][0, 3] += 0.5
        flame_param['jaw'][0, 0] += 0.03

    rest_neck = None
    if tracked_rest_pose and getattr(gaussians, 'flame_param', None) is not None:
        tracked = gaussians.flame_param.get('neck_pose')
        if tracked is not None and len(tracked) > 0:
            rest_neck = np.median(tracked.cpu().numpy(), axis=0)

    # Neck pitched slightly down, as the viewer does for a natural look.
    demo_neck = np.array([0.1, 0.0, 0.0])
    if rest_neck is None:
        flame_param['neck'][0, 0] = demo_neck[0]
        return flame_param

    # Avatars whose tracking split the head pose between the global rotation and
    # the neck: keep the tracked neck bend (it deforms the Gaussians) and choose
    # the global rotation so the head faces the way the default pose does.
    neck_used = rest_neck + demo_neck
    rotation = (R.from_rotvec(demo_neck) * R.from_rotvec(neck_used).inv()).as_rotvec()
    flame_param['rotation'][0] = torch.from_numpy(rotation).float()
    flame_param['neck'][0] = torch.from_numpy(neck_used).float()

    # That rotation pivots around the neck root, so shift the head back to where
    # the default pose puts it.
    def mesh_center(rot, neck):
        with torch.no_grad():
            verts, _ = gaussians.flame_model(
                gaussians.flame_param['shape'][None, ...],
                flame_param['expr'].cuda(),
                torch.tensor(rot, dtype=torch.float32)[None].cuda(),
                torch.tensor(neck, dtype=torch.float32)[None].cuda(),
                flame_param['jaw'].cuda(),
                flame_param['eyes'].cuda(),
                flame_param['translation'].cuda(),
                zero_centered_at_root_node=False,
                return_landmarks=False,
                return_verts_cano=True,
                static_offset=gaussians.flame_param['static_offset'],
            )
        return verts[0].mean(dim=0).cpu()

    flame_param['translation'][0] = mesh_center(np.zeros(3), demo_neck) - mesh_center(rotation, neck_used)
    return flame_param


def render_neutral_frame(folder: Path,
                         render_size: int = DEFAULT_RENDER_SIZE,
                         radius: float = DEFAULT_RADIUS,
                         fovy: float = DEFAULT_FOVY,
                         background: Sequence[float] = DEFAULT_BACKGROUND,
                         sh_degree: int = DEFAULT_SH_DEGREE,
                         tracked_rest_pose: bool = True,
                         resting_smile: bool = True,
                         camera_json: Optional[Path] = None):
    """Render one frontal frame of an avatar's neutral face.

    Returns a PIL image of ``render_size`` squared. Raises if the avatar cannot
    be loaded, so the caller can skip it and carry on.
    """
    # Imported here so --help and the import of this module work without CUDA.
    import torch
    from gaussian_renderer import FlameGaussianModel, GaussianModel, render
    from utils.viewer_utils import OrbitCamera
    from PIL import Image

    ply_files = sorted(folder.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"no .ply in {folder}")
    # point_cloud.ply is the conventional name; otherwise take the first.
    point_path = next((p for p in ply_files if p.name == "point_cloud.ply"), ply_files[0])

    has_flame = (folder / "flame_param.npz").exists()
    gaussians = FlameGaussianModel(sh_degree) if has_flame else GaussianModel(sh_degree)
    # motion_path stays None: with has_target=False the model already picks up
    # the avatar's own flame_param.npz, and the pose is overridden below anyway.
    gaussians.load_ply(point_path, has_target=False, motion_path=None, disable_fid=[])

    if gaussians.binding is not None:
        gaussians.select_mesh_by_timestep(0)
        gaussians.update_mesh_by_param_dict(
            neutral_flame_param(gaussians, tracked_rest_pose, resting_smile))

    # A camera.json in the working directory would otherwise be picked up and
    # silently change the framing, so the path is always explicit.
    cam = OrbitCamera(render_size, render_size, r=radius, fovy=fovy,
                      convention="opencv",
                      save_path=str(camera_json) if camera_json else str(folder / ".no-camera.json"))

    class Cam:
        FoVx = float(np.radians(cam.fovx))
        FoVy = float(np.radians(cam.fovy))
        image_height = cam.image_height
        image_width = cam.image_width
        # The transposes are what the gaussian splatting rasterizer expects.
        world_view_transform = torch.tensor(cam.world_view_transform).float().cuda().T
        full_proj_transform = torch.tensor(cam.full_proj_transform).float().cuda().T
        camera_center = torch.tensor(cam.pose[:3, 3]).cuda()

    with torch.no_grad():
        rgb = render(Cam, gaussians, PipelineConfig(),
                     torch.tensor(background, dtype=torch.float32).cuda())["render"]
        frame = (rgb.clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()

    del gaussians
    torch.cuda.empty_cache()
    return Image.fromarray(frame, mode="RGB")


def avatar_folders(root: Path, names: Optional[Sequence[str]] = None) -> List[Path]:
    """Avatar folders under ``root``, or just the named ones."""
    if names:
        return [root / name for name in names]
    return [child for child in sorted(root.iterdir(), key=lambda p: p.name.lower())
            if child.is_dir() and not child.name.startswith(".") and any(child.glob("*.ply"))]


def existing_thumbnail(folder: Path) -> Optional[Path]:
    for filename in THUMBNAIL_FILENAMES:
        if (folder / filename).is_file():
            return folder / filename
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render thumbnail.jpg for the operator app's avatar picker.",
        epilog="Needs CUDA and diff_gaussian_rasterization; see doc/avatar_list.md.")
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_AVATAR_ROOT),
                        help=f"avatar directory (default: {DEFAULT_AVATAR_ROOT})")
    parser.add_argument("--avatar", action="append", metavar="NAME",
                        help="only this avatar folder; repeatable")
    parser.add_argument("--force", action="store_true",
                        help="re-render avatars that already have a thumbnail")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be rendered")
    parser.add_argument("--render-size", type=int, default=DEFAULT_RENDER_SIZE,
                        help=f"render resolution before downscaling (default: {DEFAULT_RENDER_SIZE})")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                        help=f"camera distance (default: {DEFAULT_RADIUS}, as in run.sh)")
    parser.add_argument("--fovy", type=float, default=DEFAULT_FOVY,
                        help=f"vertical field of view (default: {DEFAULT_FOVY})")
    parser.add_argument("--background", type=float, nargs=3, metavar=("R", "G", "B"),
                        default=list(DEFAULT_BACKGROUND), help="background color, 0-1")
    parser.add_argument("--sh-degree", type=int, default=DEFAULT_SH_DEGREE,
                        help=f"spherical harmonics degree (default: {DEFAULT_SH_DEGREE}, as in run.sh)")
    parser.add_argument("--no-tracked-rest-pose", action="store_true",
                        help="do not start from the median tracked head rotation and neck pose")
    parser.add_argument("--no-resting-smile", action="store_true",
                        help="render the plain zero expression instead of the viewer's resting look")
    parser.add_argument("--camera", type=Path, default=None,
                        help="camera.json to take the framing from")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"error: no such avatar directory: {args.root}", file=sys.stderr)
        return 2

    folders = avatar_folders(args.root, args.avatar)
    if not folders:
        print(f"no avatars found in {args.root}")
        return 0

    rendered = skipped = failed = 0
    for folder in folders:
        if not folder.is_dir():
            print(f"  {folder.name:<20} missing folder")
            failed += 1
            continue

        # The node leaves these out of the list, so they need no thumbnail.
        reason = check_avatar_folder(folder, args.sh_degree)
        if reason:
            print(f"  {folder.name:<20} not an avatar the viewer can load: {reason}")
            skipped += 1
            continue

        existing = existing_thumbnail(folder)
        if existing is not None and not args.force:
            print(f"  {folder.name:<20} has {existing.name} already (use --force to replace)")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  {folder.name:<20} would render")
            continue

        try:
            image = render_neutral_frame(
                folder,
                render_size=args.render_size,
                radius=args.radius,
                fovy=args.fovy,
                background=tuple(args.background),
                sh_degree=args.sh_degree,
                tracked_rest_pose=not args.no_tracked_rest_pose,
                resting_smile=not args.no_resting_smile,
                camera_json=args.camera,
            )
            data = encode_thumbnail(image)
            out_path = folder / "thumbnail.jpg"
            out_path.write_bytes(data)
            # The container runs as root; leave the file to whoever owns the avatar.
            if os.geteuid() == 0:
                st = folder.stat()
                os.chown(out_path, st.st_uid, st.st_gid)
        except Exception as exc:
            print(f"  {folder.name:<20} failed: {exc}")
            failed += 1
            continue

        rendered += 1
        print(f"  {folder.name:<20} -> {out_path.name} ({len(data)} B)")

    print(f"{rendered} rendered, {skipped} skipped, {failed} failed")
    if rendered:
        print(f"The running node picks the new thumbnails up within seconds "
              f"(it watches {args.root}) and publishes a new revision.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
