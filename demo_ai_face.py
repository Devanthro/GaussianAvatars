"""
Standalone demo of the AI face with audio-driven mouth animation.

Modes:
    python3 demo_ai_face.py --ros          Subscribe to /ai/face/mouth via ROS2
    python3 demo_ai_face.py --test-tone    Cycle synthetic vowels locally
    python3 demo_ai_face.py --live         Monitor speaker output via PipeWire
    python3 demo_ai_face.py                Default: test-tone mode

For --ros mode, run the audio node on the host first:
    python3 ros2_audio_mouth_node.py --test-tone   (or --live for real audio)
"""

import sys
import os
import time
import numpy as np
from ai_face_renderer import AIFaceRenderer
from audio_mouth_driver import AudioMouthDriver, generate_test_tone

IMAGE_PATH = '/workspace/avatars/happy_ai.png'
FALLBACK_PATH = '/home/alice/workspace/gaussian_avatar_ws/avatars/happy_ai.png'
WIDTH, HEIGHT = 768, 1024
FPS = 30


def run_demo(mode='test-tone'):
    img_path = IMAGE_PATH if os.path.exists(IMAGE_PATH) else FALLBACK_PATH
    renderer = AIFaceRenderer(img_path)

    ros_sub = None
    driver = None

    if mode == 'ros':
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        class _Sub(Node):
            def __init__(self):
                super().__init__('demo_ai_face')
                self.bs = {}
                self.create_subscription(
                    JointState, '/ai/face/mouth', self._cb, 1)

            def _cb(self, msg):
                self.bs = dict(zip(msg.name, msg.position))

        rclpy.init()
        ros_sub = _Sub()
        print("ROS2 mode: subscribing to /ai/face/mouth")
    else:
        driver = AudioMouthDriver()
        if mode == 'live':
            driver.start()
            print("Live mode: monitoring speaker output via PipeWire")
        else:
            print("Test-tone mode: cycling vowels (ah, ee, oo, eh, oh)")

    try:
        import dearpygui.dearpygui as dpg
    except ImportError:
        print("DearPyGUI not available, falling back to file output mode")
        if driver:
            _run_file_output(renderer, driver, mode == 'live')
        return

    dpg.create_context()
    dpg.create_viewport(title='AI Face Demo', width=WIDTH, height=HEIGHT,
                        decorated=False, x_pos=0, y_pos=0)
    dpg.setup_dearpygui()

    initial = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    with dpg.texture_registry():
        dpg.add_raw_texture(WIDTH, HEIGHT, initial.flatten(),
                            format=dpg.mvFormat_Float_rgb, tag='_tex')

    with dpg.window(tag='_win', no_title_bar=True, no_move=True,
                    no_resize=True, no_scrollbar=True):
        dpg.add_image('_tex', width=WIDTH, height=HEIGHT)

    dpg.set_primary_window('_win', True)
    dpg.show_viewport()

    vowels = ['ah', 'ee', 'oo', 'eh', 'oh']
    vowel_idx = 0
    phase = 0.0
    frame_dt = 1.0 / FPS

    try:
        while dpg.is_dearpygui_running():
            t0 = time.time()

            if mode == 'ros':
                import rclpy as _rclpy
                _rclpy.spin_once(ros_sub, timeout_sec=0.001)
                bs = ros_sub.bs if ros_sub.bs else {
                    'mouthSmile_L': 0.15, 'mouthSmile_R': 0.15}
            elif mode == 'test-tone':
                phase += frame_dt
                if phase < 0.5:
                    tone = generate_test_tone(vowels[vowel_idx],
                                              duration=frame_dt)
                    driver.feed_audio(tone)
                elif phase < 0.8:
                    driver.feed_audio(
                        np.zeros(int(16000 * frame_dt), dtype=np.float32))
                else:
                    phase = 0.0
                    vowel_idx = (vowel_idx + 1) % len(vowels)
                bs = driver.get_blendshapes()
            else:
                bs = driver.get_blendshapes()

            frame = renderer.render(bs, WIDTH, HEIGHT)
            dpg.set_value('_tex', frame.flatten())
            dpg.render_dearpygui_frame()

            elapsed = time.time() - t0
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    finally:
        dpg.destroy_context()
        if driver:
            driver.stop()
        if ros_sub:
            ros_sub.destroy_node()
            import rclpy as _rclpy
            _rclpy.shutdown()


def _run_file_output(renderer, driver, live):
    from PIL import Image
    vowels = ['ah', 'ee', 'oo', 'eh', 'oh']

    if live:
        driver.start()
        print("Rendering live audio to /tmp/ai_face_live.png (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(0.1)
                bs = driver.get_blendshapes()
                frame = renderer.render(bs, 640, 480)
                Image.fromarray((frame * 255).astype(np.uint8)).save(
                    '/tmp/ai_face_live.png')
        except KeyboardInterrupt:
            driver.stop()
    else:
        for vowel in vowels:
            tone = generate_test_tone(vowel, duration=0.3)
            driver.feed_audio(tone)
            bs = driver.get_blendshapes()
            frame = renderer.render(bs, 640, 480)
            path = f'/tmp/ai_face_{vowel}.png'
            Image.fromarray((frame * 255).astype(np.uint8)).save(path)
            print(f"  {vowel}: jawOpen={bs['jawOpen']:.2f}  saved {path}")

        silence = np.zeros(8000, dtype=np.float32)
        for _ in range(10):
            driver.feed_audio(silence[:800])
        bs = driver.get_blendshapes()
        frame = renderer.render(bs, 640, 480)
        Image.fromarray((frame * 255).astype(np.uint8)).save(
            '/tmp/ai_face_rest.png')
        print(f"  rest: saved /tmp/ai_face_rest.png")


if __name__ == '__main__':
    if '--ros' in sys.argv:
        run_demo('ros')
    elif '--live' in sys.argv:
        run_demo('live')
    elif '--test-tone' in sys.argv:
        run_demo('test-tone')
    else:
        run_demo('test-tone')
