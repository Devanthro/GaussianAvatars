"""
Real-time mouth animation driven by audio envelope and spectral centroid.

Monitors speaker output via PipeWire/parec, produces ARKit-compatible
mouth blendshapes with sub-10ms latency.

Pipeline: 128-sample chunks (8ms) → RMS envelope follower (instant attack,
exponential release) → spectral centroid for vowel shape → direct output.
No FFT peak-picking, no target/state smoothing layers.

Designed for NVIDIA Jetson Orin with PipeWire audio. numpy only.
"""

import numpy as np
import threading
import subprocess
import time

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 128
SILENCE_THRESHOLD = 0.004
AMPLITUDE_REF = 0.15
RELEASE_COEFF = 0.4
CENTROID_RELEASE = 0.7
BREATH_PERIOD = 4.0
BREATH_AMPLITUDE = 0.015
BREATH_ONSET_DELAY = 0.5
REST_SMILE = 0.15

# Spectral centroid boundaries for vowel mapping (Hz)
CENTROID_LOW = 400.0
CENTROID_HIGH = 2500.0
OUTPUT_SMOOTH = 0.7


class AudioMouthDriver:

    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate

        self._bs = {}
        self._init_blendshapes()
        self._lock = threading.Lock()

        self._envelope = 0.0
        self._centroid = 0.0
        self._silence_time = 0.0
        self._breath_phase = 0.0

        # Pre-compute frequency bins for centroid calculation
        self._freqs = np.fft.rfftfreq(CHUNK_SAMPLES, 1.0 / sample_rate).astype(np.float32)
        self._window = np.hanning(CHUNK_SAMPLES).astype(np.float32)

        self._capture_proc = None
        self._capture_thread = None
        self._running = False

    def _init_blendshapes(self):
        self._bs['jawOpen'] = 0.0
        self._bs['mouthFunnel'] = 0.0
        self._bs['mouthPucker'] = 0.0
        self._bs['mouthSmile_L'] = REST_SMILE
        self._bs['mouthSmile_R'] = REST_SMILE
        self._bs['mouthStretch_L'] = 0.0
        self._bs['mouthStretch_R'] = 0.0
        self._bs['mouthUpperUp_L'] = 0.0
        self._bs['mouthUpperUp_R'] = 0.0
        self._bs['mouthLowerDown_L'] = 0.0
        self._bs['mouthLowerDown_R'] = 0.0
        self._bs['jawForward'] = 0.0
        self._bs['jawLeft'] = 0.0
        self._bs['jawRight'] = 0.0
        self._bs['mouthClose'] = 0.0
        self._bs['mouthLeft'] = 0.0
        self._bs['mouthRight'] = 0.0
        self._bs['mouthFrown_L'] = 0.0
        self._bs['mouthFrown_R'] = 0.0
        self._bs['mouthDimple_L'] = 0.0
        self._bs['mouthDimple_R'] = 0.0
        self._bs['mouthRollLower'] = 0.0
        self._bs['mouthRollUpper'] = 0.0
        self._bs['mouthShrugLower'] = 0.0
        self._bs['mouthShrugUpper'] = 0.0
        self._bs['mouthPress_L'] = 0.0
        self._bs['mouthPress_R'] = 0.0

    def get_blendshapes(self):
        with self._lock:
            return dict(self._bs)

    def feed_audio(self, chunk, sample_rate=None):
        if sample_rate and sample_rate != self.sample_rate:
            ratio = self.sample_rate / sample_rate
            new_len = int(len(chunk) * ratio)
            indices = np.linspace(0, len(chunk) - 1, new_len).astype(int)
            chunk = chunk[indices]

        chunk = np.asarray(chunk, dtype=np.float32)
        if len(chunk) > 0 and np.max(np.abs(chunk)) > 2.0:
            chunk = chunk / 32768.0

        for i in range(0, len(chunk), CHUNK_SAMPLES):
            seg = chunk[i:i + CHUNK_SAMPLES]
            if len(seg) >= CHUNK_SAMPLES // 2:
                self._process(seg)

    def _process(self, chunk):
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        # Envelope follower: instant attack, exponential release
        if rms > self._envelope:
            self._envelope = rms
        else:
            self._envelope = self._envelope * RELEASE_COEFF + rms * (1.0 - RELEASE_COEFF)

        dt = len(chunk) / self.sample_rate

        with self._lock:
            if self._envelope < SILENCE_THRESHOLD:
                self._decay_to_rest()
                return

            self._silence_time = 0.0
            amplitude = min(self._envelope / AMPLITUDE_REF, 1.0)

            # Spectral centroid: weighted mean frequency of magnitude spectrum
            windowed = chunk[:CHUNK_SAMPLES] * self._window[:len(chunk)]
            mag = np.abs(np.fft.rfft(windowed, n=CHUNK_SAMPLES))
            freqs = self._freqs
            total = mag.sum()
            if total > 1e-10:
                raw_centroid = float(np.dot(freqs, mag) / total)
            else:
                raw_centroid = 800.0

            # Smooth centroid (it's noisier than envelope on small chunks)
            self._centroid = self._centroid * CENTROID_RELEASE + raw_centroid * (1.0 - CENTROID_RELEASE)

            # Map centroid to round vs wide
            # Low centroid (<800Hz) = round vowels (oo, oh)
            # High centroid (>1500Hz) = wide vowels (ee, eh)
            # Mid centroid = open vowels (ah)
            c_norm = np.clip(
                (self._centroid - CENTROID_LOW) / (CENTROID_HIGH - CENTROID_LOW),
                0.0, 1.0
            )
            round_factor = float(np.clip(1.0 - c_norm * 2.0, 0.0, 1.0)) * amplitude
            width_factor = float(np.clip(c_norm * 2.0 - 1.0, 0.0, 1.0)) * amplitude

            new_jaw = amplitude * 0.7
            target = {
                'jawOpen': new_jaw,
                'mouthFunnel': round_factor * 0.6,
                'mouthPucker': round_factor * 0.4,
                'mouthStretch_L': max(width_factor * 0.5, 0.05 * amplitude),
                'mouthStretch_R': max(width_factor * 0.5, 0.05 * amplitude),
                'mouthUpperUp_L': amplitude * 0.3,
                'mouthUpperUp_R': amplitude * 0.3,
                'mouthLowerDown_L': amplitude * 0.5,
                'mouthLowerDown_R': amplitude * 0.5,
                'mouthSmile_L': REST_SMILE * (1.0 - new_jaw),
                'mouthSmile_R': REST_SMILE * (1.0 - new_jaw),
            }
            s = OUTPUT_SMOOTH
            bs = self._bs
            for k, v in target.items():
                bs[k] += (v - bs[k]) * s

    def _decay_to_rest(self):
        bs = self._bs
        decay = 0.9
        for k in ('jawOpen', 'mouthFunnel', 'mouthPucker',
                  'mouthStretch_L', 'mouthStretch_R',
                  'mouthUpperUp_L', 'mouthUpperUp_R',
                  'mouthLowerDown_L', 'mouthLowerDown_R'):
            bs[k] *= decay
        bs['mouthSmile_L'] += (REST_SMILE - bs['mouthSmile_L']) * (1.0 - decay)
        bs['mouthSmile_R'] += (REST_SMILE - bs['mouthSmile_R']) * (1.0 - decay)

    def _breathe(self, dt):
        if self._silence_time < BREATH_ONSET_DELAY:
            return
        self._breath_phase += dt
        breath = BREATH_AMPLITUDE * np.sin(
            2.0 * np.pi * self._breath_phase / BREATH_PERIOD
        )
        self._bs['jawOpen'] = max(0.0, self._bs['jawOpen'] + breath)

    def start(self, device=None):
        if self._running:
            return
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, args=(device,), daemon=True
        )
        self._capture_thread.start()

    def stop(self):
        self._running = False
        if self._capture_proc:
            try:
                self._capture_proc.terminate()
                self._capture_proc.wait(timeout=2)
            except Exception:
                try:
                    self._capture_proc.kill()
                except Exception:
                    pass
            self._capture_proc = None
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
            self._capture_thread = None

    def _capture_loop(self, device):
        source = device or self._find_monitor_source()
        if not source:
            print("[AudioMouthDriver] No monitor source found")
            return

        print(f"[AudioMouthDriver] Capturing from: {source}")
        cmd = [
            'parec', '-d', str(source),
            '--format=s16le',
            '--rate=' + str(self.sample_rate),
            '--channels=1',
            '--latency-msec=5',
        ]

        try:
            self._capture_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("[AudioMouthDriver] parec not found")
            return

        bytes_per_chunk = CHUNK_SAMPLES * 2
        while self._running and self._capture_proc.poll() is None:
            data = self._capture_proc.stdout.read(bytes_per_chunk)
            if not data:
                break
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            self._process(samples)

    def _find_monitor_source(self):
        candidates = [
            'echo-cancel-sink.monitor',
            'alsa_output.platform-sound.analog-stereo.monitor',
        ]
        try:
            result = subprocess.run(
                ['pactl', 'list', 'sources', 'short'],
                capture_output=True, text=True, timeout=5
            )
            available = result.stdout
            for c in candidates:
                if c in available:
                    print(f"[AudioMouthDriver] Found: {c}")
                    return c
            for line in available.strip().split('\n'):
                if '.monitor' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        print(f"[AudioMouthDriver] Found: {parts[1]}")
                        return parts[1]
        except Exception:
            pass
        return candidates[0]


def generate_test_tone(vowel, duration=0.5, sample_rate=SAMPLE_RATE):
    formants = {
        'ah': (750, 1200),
        'ee': (300, 2300),
        'oo': (300, 800),
        'eh': (500, 1800),
        'oh': (500, 900),
    }
    f1, f2 = formants.get(vowel, (500, 1500))
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    signal = np.zeros_like(t)
    f0 = 120.0
    for h in range(1, 40):
        freq = f0 * h
        g1 = np.exp(-((freq - f1) ** 2) / (2.0 * 80.0 ** 2))
        g2 = np.exp(-((freq - f2) ** 2) / (2.0 * 120.0 ** 2))
        signal += (g1 + g2 * 0.7) * np.sin(2.0 * np.pi * freq * t)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.5
    return signal.astype(np.float32)


def main_test_tone():
    driver = AudioMouthDriver()
    vowels = ['ah', 'ee', 'oo', 'eh', 'oh']

    print("Audio Mouth Driver — Envelope + Centroid (v2)")
    print("=" * 60)

    for vowel in vowels:
        tone = generate_test_tone(vowel, duration=0.3)
        driver.feed_audio(tone)
        bs = driver.get_blendshapes()
        print(f"\n  {vowel}:")
        print(f"    jawOpen      {bs['jawOpen']:.3f}")
        print(f"    mouthFunnel  {bs['mouthFunnel']:.3f}")
        print(f"    mouthPucker  {bs['mouthPucker']:.3f}")
        print(f"    mouthStretch {bs['mouthStretch_L']:.3f}")
        print(f"    mouthSmile   {bs['mouthSmile_L']:.3f}")
        print(f"    lowerLipDown {bs['mouthLowerDown_L']:.3f}")

    print(f"\n--- Silence (500ms) ---")
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    for _ in range(10):
        driver.feed_audio(silence[:SAMPLE_RATE // 20])
    bs = driver.get_blendshapes()
    print(f"    jawOpen      {bs['jawOpen']:.3f}")
    print(f"    mouthSmile   {bs['mouthSmile_L']:.3f}")
    print("\nDone.")


def main_live():
    driver = AudioMouthDriver()
    driver.start()
    print("Listening to speaker output... (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(0.05)
            bs = driver.get_blendshapes()
            jaw = bs['jawOpen']
            fun = bs['mouthFunnel']
            stretch = bs['mouthStretch_L']
            bar = '#' * int(jaw * 40)
            print(f"\r jaw={jaw:.2f} fun={fun:.2f} str={stretch:.2f} [{bar:<40}]",
                  end='', flush=True)
    except KeyboardInterrupt:
        driver.stop()
        print("\nStopped.")


if __name__ == '__main__':
    import sys
    if '--test-tone' in sys.argv:
        main_test_tone()
    elif '--live' in sys.argv:
        main_live()
    else:
        print("Usage: python audio_mouth_driver.py [--test-tone|--live]")
        main_test_tone()
