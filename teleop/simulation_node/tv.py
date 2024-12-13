import numpy as np
from simulation_img import run
from teleop.TeleVision import OpenTeleVision
from teleop.Preprocessor import VuerPreprocessorLegacy as VuerPreprocessor
from teleop.constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig

from pathlib import Path
import yaml
from multiprocessing import   shared_memory, Queue,  Event


class VuerTeleop:
    def __init__(self, config_file_path):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        # self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=True)
        self.processor = VuerPreprocessor()

        # RetargetingConfig.set_default_urdf_dir('../assets')
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
        ]
        )

        left_wrist_mat[:3, :3] =  left_wrist_mat[:3, :3] @ rotation_matrix_y_90_min
        right_wrist_mat[:3, :3] = right_wrist_mat[:3, :3] @ rotation_matrix_y_90

        left_wrist_mat[2, 3] += 0.45
        right_wrist_mat[2, 3] += 0.45
        # left_wrist_mat[0, 3] -= 0.20
        # right_wrist_mat[0, 3] -= 0.20
        # left_wrist_mat[:3, 3] = np.array([0,0,0], dtype=np.float32)
        # right_wrist_mat[:3, 3] = np.array([0,0,0], dtype=np.float32)
        print('left_wrist_trans', left_wrist_mat[:3, 3])
        print('right_wrist_trans', right_wrist_mat[:3, 3])

        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]

        return head_rmat, left_wrist_mat, right_wrist_mat, left_qpos, right_qpos

if __name__ == '__main__':
    teleoperator = VuerTeleop('inspire_hand.yml')
    # simulator = Sim()

    try:
        while True:
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
            # head_rmat= np.eye(3)
            # left_pose = np.eye(4)
            # right_pose = np.eye(4)
            # left_qpos = np.zeros((12,))
            # right_qpos = np.zeros((12,))
            # head_rmat, left_pose, right_pose, left_qpos, right_qpos = None, None, None, None, None
            # left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)
            left_img, right_img = run((head_rmat, left_pose, right_pose, left_qpos, right_qpos))
            np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))
            # cv2.imshow('left_img', left_img)
            # cv2.imshow('right_img', right_img)
            # cv2.waitKey(1)
    except KeyboardInterrupt:
        # simulator.end()
        exit(0)