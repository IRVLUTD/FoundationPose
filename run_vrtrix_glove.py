from estimater import *
from datareader import *
import argparse


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--video_dir', type=str, default='/home/bowen/debug/foundationpose_prep/left')
  parser.add_argument('--min_n_views', type=int, default=40)
  parser.add_argument('--inplane_step', type=float, default=60)
  parser.add_argument('--cluster_rot_diff', type=float, default=30)
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--zfar', type=float, default=np.inf)
  parser.add_argument('--debug_dir', type=str, default='/home/bowen/debug/foundationpose_prep/left_debug')
  parser.add_argument('--out_dir', type=str, default='/home/bowen/debug/refined_hand')
  parser.add_argument('--debug', type=int, default=2)
  parser.add_argument('--max_frames', type=int, default=-1, help='Max frames to process, -1 for all')
  parser.add_argument('--frame_ids', type=str, default=None, help='Comma-separated frame IDs to process, e.g. frame_000240')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  video_dir = args.video_dir
  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir} && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  K = np.loadtxt(f'{video_dir}/cam_K.txt').reshape(3,3)

  color_files = sorted(glob.glob(f'{video_dir}/rgb/*.png') + glob.glob(f'{video_dir}/rgb/*.jpg'))
  logging.info(f'Found {len(color_files)} color frames')

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()

  est = None

  if args.frame_ids is not None:
    frame_ids = set(args.frame_ids.split(','))
    color_files = [f for f in color_files if os.path.basename(f).replace('.png','').replace('.jpg','') in frame_ids]
  elif args.max_frames > 0:
    color_files = color_files[:args.max_frames]

  for i, color_file in enumerate(color_files):
    id_str = os.path.basename(color_file).replace('.png','').replace('.jpg','')
    logging.info(f'i:{i}, id_str:{id_str}')

    color = imageio.imread(color_file)[...,:3]
    H, W = color.shape[:2]

    depth_file = f'{video_dir}/depths/{id_str}.png'
    depth = cv2.imread(depth_file, -1) / 1e3
    depth[(depth < 0.001) | (depth >= args.zfar)] = 0

    mask_file = f'{video_dir}/masks/{id_str}.png'
    mask = cv2.imread(mask_file, -1)
    if len(mask.shape) == 3:
      for c in range(3):
        if mask[..., c].sum() > 0:
          mask = mask[..., c]
          break
    mask = mask.astype(bool)

    # Zero out background pixels using the segmentation mask
    color[~mask] = 0
    depth[~mask] = 0

    mesh_file = f'{video_dir}/meshes/{id_str}.obj'
    if not os.path.exists(mesh_file):
      logging.info(f'Mesh file {mesh_file} not found, skipping frame')
      continue
    mesh = trimesh.load(mesh_file, skip_materials=False, force='mesh')

    init_pose_file = f'{video_dir}/init_poses/{id_str}.txt'
    if os.path.exists(init_pose_file):
      init_pose = np.loadtxt(init_pose_file).reshape(4,4)
      mesh.apply_transform(init_pose)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    if est is None:
      est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, viewpoint_predictor=None, min_n_views=args.min_n_views, inplane_step=args.inplane_step, cluster_rot_diff=args.cluster_rot_diff, debug_dir=debug_dir, debug=debug)
    else:
      est.reset_object(mesh.vertices, mesh.vertex_normals, mesh=mesh)

    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

    # Save per-frame results
    m = mesh.copy()
    m.apply_transform(pose)
    m.export(f'{debug_dir}/model_tf.obj')
    np.savetxt(f'{debug_dir}/ob_in_cam.txt', pose.reshape(4,4))

    if debug >= 2:
      xyz_map = depth2xyzmap(depth, K)
      valid = depth >= 0.001
      pcd = toOpen3dCloud(xyz_map[valid], color[valid])
      o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)

    if debug >= 1:
      center_pose = pose @ np.linalg.inv(to_origin)
      vis = render_overlay(pose, mesh=mesh, color=color, K=K, H=H, W=W)
      vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)

    if debug >= 2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{id_str}.png', vis)
