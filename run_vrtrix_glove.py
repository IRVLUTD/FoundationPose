from estimater import *
from datareader import *
import argparse
import os
import glob
import logging
import imageio
import cv2
import numpy as np
import trimesh
from pathlib import Path


def render_overlay(
    pose,
    mesh,
    color,
    K,
    H,
    W,
    overlay_color=(0, 255, 0),
    alpha=0.4,
    draw_edges=True,
):
    """
    Render a simple mesh overlay onto an RGB image using OpenCV.
    """
    vis = color.copy()
    if vis.dtype != np.uint8:
        vis = np.clip(vis, 0, 255).astype(np.uint8)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if len(verts) == 0 or len(faces) == 0:
        return vis

    # transform vertices to camera coordinates
    verts_h = np.concatenate([verts, np.ones((len(verts), 1), dtype=np.float32)], axis=1)
    verts_cam = (pose @ verts_h.T).T[:, :3]

    z = verts_cam[:, 2]
    valid_v = z > 1e-6
    if valid_v.sum() == 0:
        return vis

    # project vertices
    proj = (K @ verts_cam.T).T
    uv = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)

    overlay = vis.copy()

    # painter’s algorithm
    face_z = z[faces].mean(axis=1)
    valid_faces = np.all(z[faces] > 1e-6, axis=1)
    face_indices = np.where(valid_faces)[0]
    face_indices = face_indices[np.argsort(face_z[face_indices])[::-1]]

    for fi in face_indices:
        tri = uv[faces[fi]].astype(np.int32)

        if np.any(np.isnan(tri)) or np.any(np.isinf(tri)):
            continue

        min_xy = tri.min(axis=0)
        max_xy = tri.max(axis=0)
        if max_xy[0] < 0 or max_xy[1] < 0 or min_xy[0] >= W or min_xy[1] >= H:
            continue

        cv2.fillConvexPoly(overlay, tri, overlay_color, lineType=cv2.LINE_AA)

        if draw_edges:
            cv2.polylines(
                overlay,
                [tri],
                isClosed=True,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

    return vis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='data/foundationpose_prep/left')
    parser.add_argument('--min_n_views', type=int, default=40)
    parser.add_argument('--inplane_step', type=float, default=60)
    parser.add_argument('--cluster_rot_diff', type=float, default=30)
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--zfar', type=float, default=np.inf)
    parser.add_argument('--debug_dir', type=str, default='left_debug')
    parser.add_argument('--out_pose_dir', type=str, default='refined_poses')
    parser.add_argument('--out_image_dir', type=str, default='refined_images')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--max_frames', type=int, default=-1)
    parser.add_argument('--frame_ids', type=str, default=None)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    video_dir = Path(args.video_dir)
    out_pose_dir = video_dir / args.out_pose_dir
    out_image_dir = video_dir / args.out_image_dir
    debug_dir = video_dir / ".." / args.debug_dir    
    debug = args.debug
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_pose_dir.mkdir(parents=True, exist_ok=True)
    out_image_dir.mkdir(parents=True, exist_ok=True)

    K = np.loadtxt(f'{video_dir}/cam_K_oak.txt').reshape(3, 3)

    color_files = sorted(
        glob.glob(f'{video_dir}/rgb_oak/*.png') +
        glob.glob(f'{video_dir}/rgb_oak/*.jpg')
    )

    logging.info(f'Found {len(color_files)} color frames')

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    est = None

    if args.frame_ids is not None:
        frame_ids = set(args.frame_ids.split(','))
        color_files = [
            f for f in color_files
            if os.path.basename(f).replace('.png', '').replace('.jpg', '') in frame_ids
        ]
    elif args.max_frames > 0:
        color_files = color_files[:args.max_frames]

    for i, color_file in enumerate(color_files):

        id_str = os.path.basename(color_file).replace('.png', '').replace('.jpg', '')
        logging.info(f'i:{i}, id_str:{id_str}')

        color = imageio.imread(color_file)[..., :3]
        H, W = color.shape[:2]

        depth_file = f'{video_dir}/depths_oak/{id_str}.png'
        depth = cv2.imread(depth_file, -1) / 1e3
        depth[(depth < 0.001) | (depth >= args.zfar)] = 0

        mask_file = f'{video_dir}/masks_oak/{id_str}.png'
        mask = cv2.imread(mask_file, -1)

        if len(mask.shape) == 3:
            for c in range(3):
                if mask[..., c].sum() > 0:
                    mask = mask[..., c]
                    break
        mask = mask.astype(bool)

        # Zero out background pixels using the segmentation mask
        color_origin = color.copy()
        color[~mask] = 0
        depth[~mask] = 0

        mesh_file = f'{video_dir}/meshes/{id_str}.obj'
        if not os.path.exists(mesh_file):
            logging.info(f'Mesh file {mesh_file} not found, skipping frame')
            continue
        mesh = trimesh.load(mesh_file, skip_materials=False, force='mesh')

        init_pose_file = f'{video_dir}/init_poses_oak/{id_str}.txt'
        if os.path.exists(init_pose_file):
            init_pose = np.loadtxt(init_pose_file).reshape(4, 4)

        if est is None:
            est = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir=debug_dir,
                debug=debug
            )
        else:
            est.reset_object(mesh.vertices, mesh.vertex_normals, mesh=mesh)

        pose = est.register(
            K=K,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=args.est_refine_iter
        )            

        # pose = est.register_with_pose(
        #     init_pose = init_pose,
        #     K=K,
        #     rgb=color,
        #     depth=depth,
        #     ob_mask=mask,
        #     iteration=args.est_refine_iter
        # )

        # save pose
        np.savetxt(f'{out_pose_dir}/{id_str}.txt', pose.reshape(4, 4))

        # save image
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        center_pose = pose @ np.linalg.inv(to_origin)

        vis = render_overlay(
            pose,
            mesh=mesh,
            color=color_origin,
            K=K,
            H=H,
            W=W
        )

        vis = draw_xyz_axis(
            vis,
            ob_in_cam=center_pose,
            scale=0.1,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True
        )
        imageio.imwrite(f'{out_image_dir}/{id_str}.png', vis) 

        if debug > 0:
            cv2.imshow('1', vis[..., ::-1])
            cv2.waitKey()

        if debug >= 2:
            xyz_map = depth2xyzmap(depth, K)
            valid = depth >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)

            # save the initial pose
            vis = render_overlay(
                init_pose,
                mesh=mesh,
                color=color_origin,
                K=K,
                H=H,
                W=W
            )
            imageio.imwrite(f'{debug_dir}/init_pose.png', vis) 