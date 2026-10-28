"""
ROS2 node for audio-driven mouth animation.

Runs on the HOST (not inside Docker) to access PipeWire audio.
Captures speaker output via echo-cancel-sink monitor, extracts
formant features, and publishes mouth blendshapes for the AI face.

Usage:
    # Live PipeWire capture (default)
    python3 ros2_audio_mouth_node.py

    # Test tone mode — cycles synthetic vowels
    python3 ros2_audio_mouth_node.py --test-tone

    # Specify PipeWire monitor source
    python3 ros2_audio_mouth_node.py --device echo-cancel-sink.monitor
"""

import sys
import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from audio_mouth_driver import AudioMouthDriver, generate_test_tone


class AudioMouthNode(Node):

    def __init__(self, device=None, test_tone=False):
        super().__init__('audio_mouth_node')
        self.driver = AudioMouthDriver()
        self._test_tone = test_tone
        self._device = device

        self.mouth_pub = self.create_publisher(JointState, '/ai/face/mouth', 1)
        self.active_pub = self.create_publisher(Bool, '/ai/face/active', 1)

        self._publish_active(True)

        if test_tone:
            self.get_logger().info('Test-tone mode: cycling synthetic vowels')
            self._tone_vowels = ['ah', 'ee', 'oo', 'eh', 'oh']
            self._tone_idx = 0
            self._tone_phase = 0.0
            self.create_timer(1.0 / 30.0, self._timer_test_tone)
        else:
            self.driver.start(device=device)
            self.get_logger().info(f'PipeWire capture started (device={device or "auto"})')
            self.create_timer(1.0 / 30.0, self._timer_live)

    def _publish_active(self, active):
        msg = Bool()
        msg.data = active
        self.active_pub.publish(msg)

    def _timer_live(self):
        bs = self.driver.get_blendshapes()
        self._publish_mouth(bs)

    def _timer_test_tone(self):
        self._tone_phase += 1.0 / 30.0

        if self._tone_phase < 0.6:
            vowel = self._tone_vowels[self._tone_idx]
            tone = generate_test_tone(vowel, duration=1.0 / 30.0)
            self.driver.feed_audio(tone)
        elif self._tone_phase < 1.0:
            silence = np.zeros(int(16000 / 30), dtype=np.float32)
            self.driver.feed_audio(silence)
        else:
            self._tone_phase = 0.0
            self._tone_idx = (self._tone_idx + 1) % len(self._tone_vowels)
            vowel = self._tone_vowels[self._tone_idx]
            self.get_logger().info(f'Vowel: {vowel}')

        bs = self.driver.get_blendshapes()
        self._publish_mouth(bs)

    def _publish_mouth(self, blendshapes):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(blendshapes.keys())
        msg.position = [float(v) for v in blendshapes.values()]
        self.mouth_pub.publish(msg)

    def destroy_node(self):
        self._publish_active(False)
        self.driver.stop()
        super().destroy_node()


def main():
    device = None
    test_tone = False

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--test-tone':
            test_tone = True
        elif arg == '--device' and i + 1 < len(args):
            device = args[i + 1]

    rclpy.init()
    node = AudioMouthNode(device=device, test_tone=test_tone)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
