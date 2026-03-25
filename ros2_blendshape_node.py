import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import  Bool, String
import time
import traceback
import subprocess
import os

AVATAR_PATH = "/workspace/avatars/"
IDLE_VIDEO_PATH = os.path.expanduser("/workspace/avatars/robody-sleeping.mov")

OVR_ARKIT_BLENDSHAPES_MAP = {
    "19": ("jawOpen", 1.7),
    "50": ("cheekPuff", 1.0),  # Note: Cheek Suck L/R in old naming
    "88": ("browInnerUp", 1.0),  # Note: Used for both L/R in old naming
    "89": ("browDown_L", 1.0),
    "90": ("browDown_R", 1.0),
    "91": ("browOuterUp_L", 1.0),
    "92": ("browOuterUp_R", 1.0),
    "93": ("eyeLookUp_L", 1.0),
    "94": ("eyeLookUp_R", 1.0),
    "95": ("eyeLookDown_L", 1.0),
    "96": ("eyeLookDown_R", 1.0),
    "97": ("eyeLookIn_L", 1.0),
    "98": ("eyeLookOut_L", 1.0),
    "99": ("eyeLookIn_R", 1.0),
    "100": ("eyeLookOut_R", 1.0),
    "101": ("eyeBlink_L", 1.0),
    "102": ("eyeBlink_R", 1.0),
    "103": ("eyeSquint_L", 1.0),
    "104": ("eyeSquint_R", 1.0),
    "105": ("eyeWide_L", 1.0),
    "106": ("eyeWide_R", 1.0),
    "107": ("cheekPuff", 1.0),  # Note: Used for both L/R in old naming
    "108": ("cheekSquint_L", 1.0),
    "109": ("cheekSquint_R", 1.0),
    "110": ("noseSneer_L", 1.0),
    "111": ("noseSneer_R", 1.0),
    "113": ("jawForward", 1.0),
    "114": ("jawLeft", 1.0),
    "115": ("jawRight", 1.0),
    "116": ("mouthFunnel", 2.0),  # Note: Used for all funnel variants in old naming
    "117": ("mouthPucker", 2.0),  # Note: Used for both L/R in old naming
    "118": ("mouthLeft", 1.3),
    "119": ("mouthRight", 1.3),
    "120": ("mouthRollUpper", 1.0),  # Note: Lip Suck LT/RT in old naming
    "121": ("mouthRollLower", 1.0),  # Note: Lip Suck LB/RB in old naming
    "122": ("mouthShrugUpper", 1),
    "123": ("mouthShrugLower", 1),
    "124": ("mouthClose", 1.0),
    "125": ("mouthSmile_L", 1.5),
    "126": ("mouthSmile_R", 1.5),
    "127": ("mouthFrown_L", 1),
    "128": ("mouthFrown_R", 1),
    "129": ("mouthDimple_L", 1),
    "130": ("mouthDimple_R", 1),
    "131": ("mouthUpperUp_L", 1),
    "132": ("mouthUpperUp_R", 1),
    "133": ("mouthLowerDown_L", 1),
    "134": ("mouthLowerDown_R", 1),
    "135": ("mouthPress_L", 1),
    "136": ("mouthPress_R", 1),
    "137": ("mouthStretch_L", 1.5),
    "138": ("mouthStretch_R", 1.5)
}

NAME_MAP = {
    "raf": "rafael3",
    "alona": "alona5",
    "caya": "tatjana2",
    "female": "simone.png",
    "male": "mark1",
    "johannes": "hannes",
    "juri": "juri"
}

# OVR_ARKIT_BLENDSHAPES_MAP = {
#     "19": "jawOpen",
#     "50": "cheekPuff",  # Note: Cheek Suck L/R in old naming
#     "88": "browInnerUp",  # Note: Used for both L/R in old naming
#     "89": "browDown_L",
#     "90": "browDown_R",
#     "91": "browOuterUp_L",
#     "92": "browOuterUp_R",
#     "93": "eyeLookUp_L",
#     "94": "eyeLookUp_R",
#     "95": "eyeLookDown_L",
#     "96": "eyeLookDown_R",
#     "97": "eyeLookIn_L",
#     "98": "eyeLookOut_L",
#     "99": "eyeLookIn_R",
#     "100": "eyeLookOut_R",
#     "101": "eyeBlink_L",
#     "102": "eyeBlink_R",
#     "103": "eyeSquint_L",
#     "104": "eyeSquint_R",
#     "105": "eyeWide_L",
#     "106": "eyeWide_R",
#     "107": "cheekPuff",  # Note: Used for both L/R in old naming
#     "108": "cheekSquint_L",
#     "109": "cheekSquint_R",
#     "110": "noseSneer_L",
#     "111": "noseSneer_R",
#     "113": "jawForward",
#     "114": "jawLeft",
#     "115": "jawRight",
#     "116": "mouthFunnel",  # Note: Used for all funnel variants in old naming
#     "117": "mouthPucker",  # Note: Used for both L/R in old naming
#     "118": "mouthLeft",
#     "119": "mouthRight",
#     "120": "mouthRollUpper",  # Note: Lip Suck LT/RT in old naming
#     "121": "mouthRollLower",  # Note: Lip Suck LB/RB in old naming
#     "122": "mouthShrugUpper",
#     "123": "mouthShrugLower",
#     "124": "mouthClose",
#     "125": "mouthSmile_L",
#     "126": "mouthSmile_R",
#     "127": "mouthFrown_L",
#     "128": "mouthFrown_R",
#     "129": "mouthDimple_L",
#     "130": "mouthDimple_R",
#     "131": "mouthUpperUp_L",
#     "132": "mouthUpperUp_R",
#     "133": "mouthLowerDown_L",
#     "134": "mouthLowerDown_R",
#     "135": "mouthPress_L",
#     "136": "mouthPress_R",
#     "137": "mouthStretch_L",
#     "138": "mouthStretch_R"
# }

class ROS2Subscriber(Node):
    def __init__(self, viewer):
        super().__init__('blendshape_subscriber')
        self.viewer = viewer

        self.blendshape_data = {}
        self.eyes_data = None

        # Initialize time tracking
        self.last_msg_time = self.get_clock().now()

        # Image display state
        self.current_name = None
        self.image_path = ""
        self.presence = False
        self.eog_process = None
        self.ffplay_process = None
        
        self.subscription = self.create_subscription(
            JointState,
            '/operator/face/expressions',
            self.expressions_callback,
            1
        )

        self.presence_subscription = self.create_subscription(
            Bool,
            '/operator/presence',
            self.presence_callback,
            1
        )

        self.name_subscription = self.create_subscription(
            String,
            '/operator/name',
            self.name_callback,
            1)

        self.eyes_subscription = self.create_subscription(
            PoseStamped,
            '/operator/eyes',
            self.eyes_callback,
            1
        )

        # self.create_timer(1/8.0, self.timer_callback)

        # Start idle video by default (no operator present on startup)
        self.show_idle_video()

    def timer_callback(self):

        if self.eyes_data is not None:
            self.viewer.update_eyes_from_ros(self.eyes_data)
        
        if self.blendshape_data is not None and self.blendshape_data:
            self.viewer.update_blendshapes_from_ros(self.blendshape_data)

    def presence_callback(self, msg):
        self.presence = msg.data
        # Image mode: don't show splatting, use eog instead
        if self.image_path.endswith('.png'):
            self.viewer.splatting_visible = False
        else:
            self.viewer.splatting_visible = msg.data
        self.update_image_display()
        # Show idle video when no operator is present
        if not msg.data:
            self.show_idle_video()
        else:
            self.hide_idle_video()
        # self.viewer.toggle_splatting(msg.data)

    def name_callback(self, msg):
        self.current_name = msg.data.strip().lower()
        self.get_logger().info(f"Name callback: {self.current_name}, presence: {self.presence}")

        if self.current_name not in NAME_MAP:
            self.get_logger().info(f"Avatar name not found: {self.current_name}")
            return

        avatar_value = NAME_MAP[self.current_name]
        self.image_path = f"{AVATAR_PATH}/{avatar_value}"
        self.get_logger().info(f"Avatar value: {avatar_value}, image_path: {self.image_path}")

        if avatar_value.endswith('.png'):
            # Image mode - hide splatting, show image via eog
            self.viewer.splatting_visible = False
            self.update_image_display()
        else:
            # Gaussian avatar mode - hide any image, load avatar
            self.hide_image()
            self.viewer.avatar_path = self.image_path
            self.viewer.splatting_visible = self.presence
        # self.viewer.unload_avatar()
        # self.viewer.load_avatar(f"{AVATAR_PATH}/{msg.data}")

    def expressions_callback(self, msg):
        # current_time = self.get_clock().now()
        # if (current_time - self.last_msg_time).nanoseconds / 1e9 >= self.desired_period:
        # self.get_logger().info('Received blendshape message:', throttle_duration_sec=1)
        self.blendshape_data.clear()
        for index, value in zip(msg.name, msg.position):
            mapping = OVR_ARKIT_BLENDSHAPES_MAP.get(index, (index, 1.0))  # Default weight is 1.0
            blendshape_name, weight = mapping
            if "Look" in blendshape_name:
                continue
            normalized_value = (value/100.0) * weight
            self.blendshape_data[blendshape_name] = normalized_value
        # self.viewer.update_blendshapes_from_ros(self.blendshape_data)
        # self.get_logger().info(f"{list(self.blendshape_data.keys())[0]}: {list(self.blendshape_data.values())[0]}", throttle_duration_sec=10)
        # self.last_msg_time = current_time

    def eyes_callback(self, msg):
        self.eyes_data = [msg.pose.orientation.x / 100.0, msg.pose.orientation.y / 100.0]
        # self.get_logger().info(f"Eyes: {self.eyes_data}", throttle_duration_sec=10)

    def update_image_display(self):
        if self.image_path.endswith('.png') and self.presence:
            self.show_image(self.image_path)
        else:
            self.hide_image()

    def show_image(self, path):
        if self.eog_process is not None and self.eog_process.poll() is None:
            return
        try:
            env = os.environ.copy()
            env['DISPLAY'] = ':0'
            # image_path = os.path.expanduser(f'{AVATAR_PATH}/mark.png')
            image_path = path
            self.eog_process = subprocess.Popen(
                ['eog', '--fullscreen', image_path],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.get_logger().info(f'Displaying image: {image_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to display image: {e}')

    def hide_image(self):
        if self.eog_process is not None and self.eog_process.poll() is None:
            try:
                self.eog_process.terminate()
                self.eog_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.eog_process.kill()
            except Exception as e:
                self.get_logger().error(f'Failed to hide image: {e}')
            finally:
                self.eog_process = None

    def show_idle_video(self):
        if self.ffplay_process is not None and self.ffplay_process.poll() is None:
            return  # already running
        try:
            env = os.environ.copy()
            env['DISPLAY'] = ':0'
            self.ffplay_process = subprocess.Popen(
                ['ffplay', '-an', '-vf', 'setpts=7*PTS', '-fs', '-loop', '0', IDLE_VIDEO_PATH],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.get_logger().info(f'Playing idle video: {IDLE_VIDEO_PATH}')
        except Exception as e:
            self.get_logger().error(f'Failed to play idle video: {e}')

    def hide_idle_video(self):
        if self.ffplay_process is not None and self.ffplay_process.poll() is None:
            try:
                self.ffplay_process.terminate()
                self.ffplay_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.ffplay_process.kill()
            except Exception as e:
                self.get_logger().error(f'Failed to stop idle video: {e}')
            finally:
                self.ffplay_process = None

    # def expressions_callback(self, msg):
    #     self.get_logger().info('Received blendshape message:', throttle_duration_sec=1)
    #     self.blendshape_data = {}
    #     for index, value in zip(msg.name, msg.position):
    #         mapping = OVR_ARKIT_BLENDSHAPES_MAP.get(index, (index, 1.0))  # Default weight is 1.0
    #         blendshape_name, weight = mapping
    #         normalized_value = (value/100.0) * weight
    #         self.blendshape_data[blendshape_name] = normalized_value

def main(viewer):
    rclpy.init()
    subscriber = ROS2Subscriber(viewer)
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()
    # print("ROS2 node starting...")
    # try:
    #     # Force a clean rclpy state
    #     try:
    #         rclpy.shutdown()
    #     except:
    #         pass
            
    #     # Initialize with explicit domain ID
    #     rclpy.init()
    #     print("rclpy initialized")
        
    #     # Create the node without context parameter
    #     node = ROS2Subscriber(viewer)
    #     print("ROS2 subscriber node created")
        
    #     # Use a more basic approach to spinning
    #     try:
    #         print("Starting manual node spinning")
    #         while True:
    #             rclpy.spin_once(node, timeout_sec=0.1)
    #             time.sleep(0.01)
    #     except KeyboardInterrupt:
    #         pass
    #     except Exception as e:
    #         print(f"Error spinning node: {e}")
    #     finally:
    #         node.destroy_node()
    #         rclpy.shutdown()
    # except Exception as e:
    #     print(f"Error in ROS2 initialization: {e}")
    #     traceback.print_exc()
