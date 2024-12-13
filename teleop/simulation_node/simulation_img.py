
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
from teleop.robot_control.robot_arm_ik_wht import Arm_IK


class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        file_name = file_name
        flip_visual_attachments = flip_visual_attachments



# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {"name": "--show_axis", "action": "store_false", "help": "Visualize DOF axis"}])



# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = False
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)


# load asset
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.use_mesh_materials = True
asset_root = "../../assets"
robot_asset_path = "fdr_IMU_09052/urdf/fdr_upper_un_limit_hand.urdf"
asset = gym.load_asset(sim, asset_root, robot_asset_path, asset_options)

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)
rigid_names = gym.get_asset_rigid_body_names(asset)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states['pos']

# set up the env grid
num_envs = 1
num_per_row = 6
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# cache useful handles
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.32)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)
    
# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()


# position the camera
cam_pos = gymapi.Vec3(1, 0., 1.3)
cam_target = gymapi.Vec3(0, 0., 1.3)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

cam_lookat_offset = np.array([1, 0, 0])
left_cam_offset = np.array([0, 0.033, 0])
right_cam_offset = np.array([0, -0.033, 0])
# cam_pos = np.array([-0.6, 0, 1.6])
cam_pos = np.array([-0.6, 0, 1.8])

# create left 1st preson viewer
camera_props = gymapi.CameraProperties()
camera_props.width = 1280
camera_props.height = 720
left_camera_handle = gym.create_camera_sensor(envs[0], camera_props)
gym.set_camera_location(left_camera_handle,
                             envs[0],
                             gymapi.Vec3(*(cam_pos + left_cam_offset)),
                             gymapi.Vec3(*(cam_pos + left_cam_offset + cam_lookat_offset)))

# create right 1st preson viewer
camera_props = gymapi.CameraProperties()
camera_props.width = 1280
camera_props.height = 720
right_camera_handle = gym.create_camera_sensor(envs[0], camera_props)
gym.set_camera_location(right_camera_handle,
                             envs[0],
                             gymapi.Vec3(*(cam_pos + right_cam_offset)),
                             gymapi.Vec3(*(cam_pos + right_cam_offset + cam_lookat_offset)))


rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)
_rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
robot_dof_states = gym.acquire_dof_state_tensor(sim)
_robot_dof_state = gymtorch.wrap_tensor(robot_dof_states)



def _draw_debug_vis(pos_joint, color_x=1, color_y=0.651, color_z=0):
    """ Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
    """
    # import ipdb; ipdb.set_trace()
    for env_id in range(num_envs):
        color_inner = [color_x, color_y, color_z]
        color_inner = tuple(color_inner)
        sphere_geom_marker = gymutil.WireframeSphereGeometry(0.04, 20, 20, None, color=color_inner)
        sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
        gymutil.draw_lines(sphere_geom_marker, gym, viewer, envs[env_id], sphere_pose)


arm_ik = Arm_IK()

arm_dof_name = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]
arm_dof_indices = []
for name in arm_dof_name:
    arm_dof_indices.append(gym.find_actor_dof_index(envs[0], actor_handles[0], name, gymapi.DOMAIN_SIM))
arm_dof_indices = np.array(arm_dof_indices)



def run(vr_mat, gym=gym):
    global i ,j
    global _rigid_body_state
    global _robot_dof_state
    global arm_dof_indices
    global arm_ik

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 提取位置和方向
    _rigid_body_state = _rigid_body_state.view(-1, 13)
    _rigid_body_pos = _rigid_body_state[..., 0:3]
    _rigid_body_orient = _rigid_body_state[..., 3:7]

    gym.refresh_rigid_body_state_tensor(sim)
    # add 13
    left_shoulder_pos, right_shoulder_pos = _rigid_body_pos[17, :], _rigid_body_pos[37, :]
    left_elbow_pos, right_elbow_pos = _rigid_body_pos[19, :], _rigid_body_pos[39, :]
    left_wrist_roll_pos, right_wrist_roll_pos = _rigid_body_pos[21, :], _rigid_body_pos[41, :]
    left_wrist_pitch_pos, right_wrist_pitch_pos = _rigid_body_pos[22, :], _rigid_body_pos[42, :]

    head_rmat, left_pose, right_pose, left_qpos, right_qpos = vr_mat
    arm_pos, arm_vel = _robot_dof_state[arm_dof_indices, 0], _robot_dof_state[arm_dof_indices, 1]
    arm_pos_np, arm_vel_np = arm_pos.numpy(), arm_vel.cpu().numpy()
    sol_q, tau_ff, flag = arm_ik.ik_fun(left_pose, right_pose, arm_pos_np, arm_vel_np)

    joint_state = np.zeros((len(dof_names), ))
    joint_state[0:7] = sol_q[0:7]
    joint_state[7:19] = left_qpos
    joint_state[19:26] = sol_q[7:14]
    joint_state[26:38] = right_qpos
    # joint_state= np.array([ 1.10445, 0.48431, 0.0924 , 0.0299 , 0.06956, -1.07944, -0.11418, 0.     , 0.     ,  0.     ,  0.     ,  0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,  0.     , 2.26669, 0.38963,  0.39287,  1.80969,  0.08264,  0.0153 ,0.48302,0.     , 0.     , 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ])
    for j in range(len(joint_state)):
        dof_positions[j] = joint_state[j]


    # clone actor state in all of the environments
    for k in range(num_envs):
        gym.set_actor_dof_states(envs[k], actor_handles[k], dof_states, gymapi.STATE_POS)

    left_pose, right_pose = left_pose.copy(), right_pose.copy()
    left_pose[:3, 3] += _rigid_body_pos[0].cpu().numpy()
    right_pose[:3, 3] += _rigid_body_pos[0].cpu().numpy()
    gym.clear_lines(viewer)
    debug_pos = []
    debug_pos.extend([left_pose[:3, 3], right_pose[:3, 3]])
    left_hand_x, right_hand_x = left_pose @ np.array([0.1, 0, 0, 1]), right_pose @ np.array([0.1, 0, 0, 1])
    debug_pos.extend([left_hand_x[:3], right_hand_x[:3]])
    left_hand_z, right_hand_z = left_pose @ np.array([0, 0, -0.1, 1]), right_pose @ np.array([0, 0, -0.1, 1])
    debug_pos.extend([left_hand_z[:3], right_hand_z[:3]])
    for i, pos in enumerate(debug_pos):
        if i < 2:
            _draw_debug_vis(pos, color_x=1, color_y=0, color_z=0)
        if 1 < i < 4:
            _draw_debug_vis(pos, color_x=0, color_y=1, color_z=0)
        if 3 < i < 8:
            _draw_debug_vis(pos, color_x=0, color_y=0, color_z=1)

    cam_pos = np.array([0, 0, 1.8])
    cam_lookat_offset = np.array([1, 0, 0])
    left_cam_offset = np.array([0, 0.033, 0])
    right_cam_offset = np.array([0, -0.033, 0])
    curr_lookat_offset = cam_lookat_offset @ head_rmat.T
    curr_left_offset = left_cam_offset @ head_rmat.T
    curr_right_offset = right_cam_offset @ head_rmat.T

    gym.set_camera_location(left_camera_handle,
                                 envs[0],
                                 gymapi.Vec3(*(cam_pos + curr_left_offset)),
                                 gymapi.Vec3(*(cam_pos + curr_left_offset + curr_lookat_offset)))
    gym.set_camera_location(right_camera_handle,
                                 envs[0],
                                 gymapi.Vec3(*(cam_pos + curr_right_offset)),
                                 gymapi.Vec3(*(cam_pos + curr_right_offset + curr_lookat_offset)))
    left_image = gym.get_camera_image(sim, envs[0], left_camera_handle, gymapi.IMAGE_COLOR)
    right_image = gym.get_camera_image(sim, envs[0], right_camera_handle, gymapi.IMAGE_COLOR)
    left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
    right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

    # update the viewer

    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    return left_image, right_image