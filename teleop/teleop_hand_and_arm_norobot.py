import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Value, Array, Lock
import threading
import logging_mp
# ===== BEGIN MULTI-CAM IMPORTS =====
from teleop.sensors.multi_camera_rs import MultiCameraRS, RSSpec, list_serials
import rerun as rr
import cv2, numpy as np
# ===== END MULTI-CAM IMPORTS =====

from teleop.carpet_tactile.sensors.sensors import MultiSensors

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from televuer import TeleVuerWrapper # NOQA E402
# from teleop.robot_control.robot_arm import G1_29_ArmController
# from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
# from teleop.robot_control.robot_hand_inspire import Inspire_Controller
# from teleop.robot_control.robot_hand_brainco import Brainco_Controller
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher # NOQA E402
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_ # NOQA E402
def publish_reset_category(category: int,publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
start_signal = True
running = True
should_toggle_recording = False
is_recording = False
def on_press(key):
    global running, start_signal, should_toggle_recording
    if key == 'r':
        start_signal = True
        logger_mp.info("Program start signal received.")
    elif key == 'q':
        stop_listening()
        running = False
    elif key == 's':
        should_toggle_recording = True
    else:
        logger_mp.info(f"{key} was pressed, but no action is defined for this key.")
listen_keyboard_thread = threading.Thread(target=listen_keyboard, kwargs={"on_press": on_press, "until": None, "sequential": False,}, daemon=True)
listen_keyboard_thread.start()

import multiprocessing as mp

mp.set_start_method('fork', force=True)  # Use spawn method for multiprocessing

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data', help = 'path to save data')
    parser.add_argument('--frequency', type = float, default = 60.0, help = 'save data\'s frequency')

    # basic control parameters
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'], help='Select end effector controller')
    # mode flags
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording')
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--carpet_tactile', action='store_true', help='Enable carpet tactile sensor data collection')
    parser.add_argument('--carpet_sensitivity', type=int, default=250, help='Set carpet tactile sensor sensitivity (default: 1.0)')
    parser.add_argument('--carpet_headless', action='store_true', help='Enable headless mode for carpet tactile sensor data collection')
    parser.add_argument('--carpet_tty', type=str, default='/dev/tty.usbserial-02857AC6', help='Set the TTY port for carpet tactile sensors (default: /dev/tty.usbserial-02857AC6)')


    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],  # Head camera resolution
        'head_camera_id_numbers': [0],
        'wrist_camera_type': 'opencv',
        'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
        'wrist_camera_id_numbers': [2, 4],
    }

    base_images = list()
    if args.carpet_tactile:
        carpet_sensor = MultiSensors([args.carpet_tty])
        logger_mp.info("initializing carpet tactile sensors...")
        carpet_sensor.init_sensors()
        logger_mp.info("initializing carpet tactile sensors...Done")

        for i in range(20):
            total_image = carpet_sensor.get()
            base_images.append(total_image)

        base_images = np.array(base_images)
        base_image = np.mean(base_images, axis=0)
        logger_mp.info("Carpet tactile sensors calibration done!")

        def get_tactile_data():
            total_image = carpet_sensor.get()
            total_image = total_image - base_image
            return total_image


    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    BINOCULAR = False

    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False
    
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

    if WRIST and args.sim:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name, server_address="127.0.0.1")
    elif WRIST and not args.sim:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVuerWrapper(binocular=BINOCULAR, use_hand_tracking=args.xr_mode == "hand", img_shape=tv_img_shape, img_shm_name=tv_img_shm.name, 
                                 return_state_data=True, return_hand_rot_data = False)

    # arm
    if args.arm == "G1_29":
        pass
        # arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        # arm_ik = G1_29_ArmIK()

    # end-effector

    if args.ee == "inspire1":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.

        # hand_ctrl = Inspire_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        hand_ctrl= None
    else:
        pass

    # simulation mode
    if args.sim:
        reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
        reset_pose_publisher.Init()
        from teleop.utils.sim_state_topic import start_sim_state_subscribe
        sim_state_subscriber = start_sim_state_subscribe()

    # controller + motion mode

    if args.record and not args.headless:
        recorder = EpisodeWriter(task_dir = args.task_dir, frequency = args.frequency, rerun_log = True)
        
    # ===== BEGIN MULTI-CAM SETUP (non-intrusive) =====
    ENABLE_EXTRA_CAMS = True   # 필요시 False로 끄기
    DEPTH_VIZ_MIN_M   = 0.25
    DEPTH_VIZ_MAX_M   = 2.00
    DEPTH_SCALE       = 0.001  # RealSense 기본값 (장비별로 다르면 조정)

    def _rr_log_rgb(path, rgb):
        if hasattr(rr, "Image"):
            rr.log(path, rr.Image(rgb))
        else:
            rr.log_image(path, rgb)

    def _depth_to_viz_rgb(depth_u16):
        dm = depth_u16.astype(np.float32) * DEPTH_SCALE
        dm = np.clip(dm, DEPTH_VIZ_MIN_M, DEPTH_VIZ_MAX_M)
        norm = ((dm - DEPTH_VIZ_MIN_M)/(DEPTH_VIZ_MAX_M-DEPTH_VIZ_MIN_M)*255.0).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    cams = None
    if ENABLE_EXTRA_CAMS:
        serials = list_serials()
        if len(serials) >= 3:
            serials = serials[:3]
        elif len(serials) == 2:
            serials = [serials[0], serials[1], "SYNTH2"]
        elif len(serials) == 1:
            serials = [serials[0], "SYNTH1", "SYNTH2"]
        else:
            serials = ["SYNTH0", "SYNTH1", "SYNTH2"]

        cam_specs = [
            RSSpec("wrist_left",  serials[0], 640, 480, 30, need_depth=False),
            RSSpec("wrist_right", serials[1], 640, 480, 30, need_depth=False),
            RSSpec("front",       serials[2], 640, 480, 30, need_depth=True),
        ]
        cams = MultiCameraRS(cam_specs)

        try:
            recorder.init_camera_dirs(recorder.task_dir, ["front_rgb","front_depth","wrist_left","wrist_right"])
        except Exception:
            pass
    # ===== END MULTI-CAM SETUP =====

    try:
        logger_mp.info("Please enter the start signal (enter 'r' to start the subsequent program)")
        # while not start_signal:
        #     time.sleep(0.01)
        # arm_ctrl.speed_gradual_max()
        while running:
            start_time = time.time()
            # ===== BEGIN MULTI-CAM LOOP =====
            if cams is not None:
                if cams.ready("wrist_left"):
                    rgb = cams.get_rgb("wrist_left")
                    _rr_log_rgb("/cams/wrist_left", rgb)
                    try: recorder.write_camera_rgb("wrist_left", rgb, 0, time.time())
                    except Exception: pass

                if cams.ready("wrist_right"):
                    rgb = cams.get_rgb("wrist_right")
                    _rr_log_rgb("/cams/wrist_right", rgb)
                    try: recorder.write_camera_rgb("wrist_right", rgb, 0, time.time())
                    except Exception: pass

                if cams.ready("front"):
                    rgb = cams.get_rgb("front")
                    d16 = cams.get_depth("front")
                    _rr_log_rgb("/cams/front_rgb", rgb)
                    if d16 is not None:
                        viz = _depth_to_viz_rgb(d16)
                        _rr_log_rgb("/cams/front_depth", viz)
                        try:
                            recorder.write_camera_rgb("front_rgb", rgb, 0, time.time())
                            recorder.write_camera_depth_u16("front_depth", d16, 0, time.time())
                        except Exception:
                            pass
            # ===== END MULTI-CAM LOOP =====

            if not args.headless:
                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                cv2.imshow("record image", tv_resized_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                    if args.sim:
                        publish_reset_category(2, reset_pose_publisher)
                elif key == ord('s'):
                    should_toggle_recording = True
                elif key == ord('a'):
                    if args.sim:
                        publish_reset_category(2, reset_pose_publisher)

            if args.record and should_toggle_recording:
                should_toggle_recording = False
                if not is_recording:
                    if recorder.create_episode():
                        is_recording = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    is_recording = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)
            # get input data
            tele_data = tv_wrapper.get_motion_state_data()
            if (args.ee == "dex3" or args.ee == "inspire1" or args.ee == "brainco") and args.xr_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            else:
                pass        

            # get current robot state data.
            # current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            # current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            # sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_arm_pose, tele_data.right_arm_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            # arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # record data
            if args.record:
                logger_mp.debug("record")
                # dex hand or gripper

                if (args.ee == "inspire1" or args.ee == "brainco") and args.xr_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []
                # head image
                current_tv_image = tv_img_array.copy()
                # wrist image
                if WRIST:
                    current_wrist_image = wrist_img_array.copy()
                # arm state and action
                # left_arm_state  = current_lr_arm_q[:7]
                # right_arm_state = current_lr_arm_q[-7:]
                # left_arm_action = sol_q[:7]
                # right_arm_action = sol_q[-7:]


                if is_recording:
                    logger_mp.debug("is_recording")
                    colors = {}
                    depths = {}
                    if BINOCULAR:
                        colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1]//2]
                        colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1]//2:]
                        if WRIST:
                            colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                            colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                    else:
                        colors[f"color_{0}"] = current_tv_image
                        if WRIST:
                            colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                            colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                    states = {
                        "left_arm": {                                                                    
                            # "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            # "qpos":   right_arm_state.tolist(),
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            # "qpos":   left_arm_action.tolist(),
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            # "qpos":   right_arm_action.tolist(),
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }

                    if args.carpet_tactile:
                        logger_mp.debug("carpet_tactile")
                        carpet_tactiles = dict()
                        tactile_data = get_tactile_data()
                        carpet_tactiles['carpet_0'] = tactile_data

                        if not args.carpet_headless:
                            tactile_render = (tactile_data / args.carpet_sensitivity) * 255
                            tactile_render = np.clip(tactile_render, 0, 255)
                            tactile_render = cv2.resize(tactile_render.astype(np.uint8), (500, 500))
                            cv2.imshow("carpet_0", tactile_render)
                            cv2.waitKey(1)

                    else:
                        carpet_tactiles = None

                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, carpet_tactiles=carpet_tactiles, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, carpet_tactiles=carpet_tactiles)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    except Exception as e:
        logger_mp.error(f"An error occurred: {e}")
        logger_mp.info("Exiting program due to an error...")
    finally:
        # ===== BEGIN MULTI-CAM CLEANUP =====
        try:
            if cams is not None:
                cams.close()
        except Exception:
            pass
        # ===== END MULTI-CAM CLEANUP =====
        # arm_ctrl.ctrl_dual_arm_go_home()
        if args.sim:
            sim_state_subscriber.stop_subscribe()
        tv_img_shm.close()
        tv_img_shm.unlink()
        if WRIST:
            wrist_img_shm.close()
            wrist_img_shm.unlink()
        if args.record:
            recorder.close()
        listen_keyboard_thread.join()
        logger_mp.info("Finally, exiting program...")
        exit(0)