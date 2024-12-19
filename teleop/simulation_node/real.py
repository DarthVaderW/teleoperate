import cv2
import numpy as np
from multiprocessing import Process, Array, shared_memory
from teleop.TeleVision import OpenTeleVision
from teleop.Preprocessor import VuerPreprocessorLegacy as VuerPreprocessor
from teleop.constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import yaml
from teleop.robot_control.robot_arm_ik_wht import Arm_IK
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.core import Qos, Policy
from teleop.simulation_node.dds_types import Angle_38, Angle_14

class VuerTeleop:
    def __init__(self, config_file_path, shm_name, img_shape):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)
        self.img_shape = img_shape
        self.img_height, self.img_width = self.resolution_cropped[:2]
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.img_array = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=True)
        self.processor = VuerPreprocessor()
        RetargetingConfig.set_default_urdf_dir('../../assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
        head_rmat = head_mat[:3, :3]
        rotation_matrix_y_90 = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        rotation_matrix_y_90_min = np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        left_wrist_mat[:3, :3] =  left_wrist_mat[:3, :3] @ rotation_matrix_y_90_min
        right_wrist_mat[:3, :3] = right_wrist_mat[:3, :3] @ rotation_matrix_y_90
        left_wrist_mat[2, 3] += 0.45
        right_wrist_mat[2, 3] += 0.45
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        return head_rmat, left_wrist_mat, right_wrist_mat, left_qpos, right_qpos

class RobotClient:
    def __init__(self):
        self.dp = DomainParticipant()
        self.command_topic = Topic(
            self.dp,
            "RobotCommand",
            Angle_38,
            qos = Qos(Policy.Reliability.Reliable(0), Policy.History.KeepLast(1))
        )
        self.command_writer = DataWriter(self.dp, self.command_topic)
        self.response_topic = Topic(
            self.dp,
            "RobotResponse",
            Angle_14,
            qos=Qos(Policy.Reliability.Reliable(0), Policy.History.KeepLast(1))
        )
        self.response_reader = DataReader(self.dp, self.response_topic)

def read_response_topic(arm_state, robot_client):
    while True:
        samples = robot_client.response_reader.take()
        for sample in samples:
            with arm_state.get_lock():  # Ensure exclusive access
                for i in range(len(arm_state)):
                    arm_state[i] = sample.data[i]

def capture_video(shm_name, img_shape):
    shm = shared_memory.SharedMemory(name=shm_name)
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        np.copyto(img_array, np.hstack((frame, frame)))

def process_teleop(teleoperator, arm_state, robot_client, arm_ik):
    while True:
        head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
        with arm_state.get_lock():
            current_state = np.array(arm_state[:])  # Copy to local numpy array for processing
        sol_q, tau_ff, flag = arm_ik.ik_fun(left_pose, right_pose, current_state, np.zeros((14,)))
        data = np.zeros((38,))
        data[:14] = sol_q
        data[14:14+12] = left_qpos
        data[14+12:14+12+12] = right_qpos
        robot_client.command_writer.write(Angle_38(data=data.tolist()))

if __name__ == '__main__':
    img_shape = (720, 2560, 3)  # Example shape
    shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    teleoperator = VuerTeleop('inspire_hand.yml', shm.name, img_shape)
    robot_client = RobotClient()
    arm_ik = Arm_IK()

    # Shared array for arm_state with 14 elements
    arm_state = Array('d', 14)  # 'd' for double precision float

    p1 = Process(target=read_response_topic, args=(arm_state, robot_client))
    p2 = Process(target=capture_video, args=(shm.name, img_shape))
    p3 = Process(target=process_teleop, args=(teleoperator, arm_state, robot_client, arm_ik))

    p1.start()
    p2.start()
    p3.start()

    try:
        p1.join()
        p2.join()
        p3.join()
    except KeyboardInterrupt:
        p1.terminate()
        p2.terminate()
        p3.terminate()
