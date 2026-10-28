"""ROS 2 interface that tells the operator app which avatars this robot has.

Three interfaces, all stock ROS 2 types, all carrying the same JSON payload
(see ``avatar_registry`` for the schema):

====================================== ===================== ==================
topic / service                        type                  direction
====================================== ===================== ==================
/robody/{robot}/avatar/list            std_srvs/Trigger      service
/robody/{robot}/avatar/list/reply      std_msgs/String       published
/robody/{robot}/avatar/list/request    std_msgs/Empty        subscribed
====================================== ===================== ==================

The reply topic is reliable + transient_local + depth 1 so a native ROS 2
subscriber that joins late still gets the last list.  ROS-TCP-Endpoint
subscribes as volatile, which is why the app also has the service and the
request topic.

Nothing is published on a timer: only on readiness, on a change of the list or
of ``active``, and once per message on the request topic.

The interface attaches to an existing node instead of creating one of its own,
so it is spun by whatever loop already spins that node (in this code base, the
viewer's render loop calls ``rclpy.spin_once`` on a single node).
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from avatar_registry import DEFAULT_AVATAR_ROOT, AvatarRegistry

#: How often the avatar directory fingerprint is checked (file-change latency).
DEFAULT_POLL_INTERVAL_S = 2.0
#: Hard upper bound between full rescans, independent of the fingerprint.
DEFAULT_RESCAN_INTERVAL_S = 20.0
#: How often the ROS thread picks up a payload prepared by the scan thread.
PUBLISH_TICK_S = 0.2

DEFAULT_ROBOT_NAME = "charlie"

#: robody-hal's environment file, written by robody-ta2-deploy (role hal). Its
#: ``ROBOT_NAME`` is the name robody-hal publishes under, /robody/<ROBOT_NAME>/.
DEFAULT_HAL_ENV_FILE = "/etc/robody-ta2/hal.env"
#: Overrides the path of that file, e.g. for tests or a different mount point.
HAL_ENV_FILE_VAR = "ROBODY_HAL_ENV_FILE"


def read_hal_robot_name(path: Optional[os.PathLike | str] = None) -> str:
    """``ROBOT_NAME`` from robody-hal's env file, or ``""`` if it has none.

    The file is ``KEY=VALUE`` per line with ``#`` comments, as Docker Compose's
    ``env_file`` reads it.  A missing or unreadable file is not an error here:
    the caller falls back to the rest of the chain.
    """
    path = Path(path or os.environ.get(HAL_ENV_FILE_VAR) or DEFAULT_HAL_ENV_FILE)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == "ROBOT_NAME":
            return value.strip().strip("'\"").strip()
    return ""


def resolve_robot_name(node=None) -> str:
    """Robot name used in the topic namespace, e.g. ``charlie``.

    Resolution order: ``ROBOT_NAME`` in robody-hal's env file
    (``/etc/robody-ta2/hal.env``, see :func:`read_hal_robot_name`) -> ROS
    parameter ``robot_name`` -> ``ROBOT_NAME`` / ``ROBODY_ROBOT_NAME`` /
    ``ROBODY_NAME`` env var -> hostname -> ``DEFAULT_ROBOT_NAME``.

    robody-hal's file comes first so the avatar list always lives under the
    same /robody/<robot>/ namespace as the rest of the robot.
    """
    value = read_hal_robot_name()
    if value:
        return value

    if node is not None:
        try:
            node.declare_parameter("robot_name", "")
            value = node.get_parameter("robot_name").get_parameter_value().string_value
            if value:
                return value.strip()
        except Exception:
            # Parameter already declared by someone else, or no parameter server.
            try:
                value = node.get_parameter("robot_name").get_parameter_value().string_value
                if value:
                    return value.strip()
            except Exception:
                pass

    for env_var in ("ROBOT_NAME", "ROBODY_ROBOT_NAME", "ROBODY_NAME"):
        value = os.environ.get(env_var, "").strip()
        if value:
            return value

    hostname = socket.gethostname().split(".")[0].strip()
    if hostname and hostname != "localhost":
        return hostname

    return DEFAULT_ROBOT_NAME


class AvatarListInterface:
    """Service + topics that expose :class:`AvatarRegistry` over ROS 2."""

    def __init__(self,
                 node,
                 avatar_root=DEFAULT_AVATAR_ROOT,
                 legacy_name_map: Optional[Dict[str, str]] = None,
                 robot_name: Optional[str] = None,
                 thumbnail_renderer: Optional[Callable[[str, Optional[Path]], object]] = None,
                 poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
                 rescan_interval_s: float = DEFAULT_RESCAN_INTERVAL_S):
        self.node = node
        self.logger = node.get_logger()
        self.robot_name = robot_name or resolve_robot_name(node)
        self.poll_interval_s = poll_interval_s
        self.rescan_interval_s = rescan_interval_s

        self.registry = AvatarRegistry(
            root=avatar_root,
            legacy_name_map=legacy_name_map,
            thumbnail_renderer=thumbnail_renderer,
            logger=self.logger,
        )

        # Only the scan thread mutates the registry; the ROS thread reads the
        # cached payload string (an atomic read) and hands over wishes through
        # the small guarded fields below.
        self._lock = threading.Lock()
        self._ready = False
        self._publish_pending = False
        #: Last id handed to set_active(); re-applied after every scan so a
        #: selection made before the first scan, or while its avatar was
        #: briefly missing, is not lost.
        self._desired_active = ""
        self._wake = threading.Event()
        self._stop = threading.Event()

        base = f"/robody/{self.robot_name}/avatar/list"
        self.service_name = base
        self.reply_topic = f"{base}/reply"
        self.request_topic = f"{base}/request"

        reply_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._reply_pub = node.create_publisher(String, self.reply_topic, reply_qos)
        self._request_sub = node.create_subscription(
            Empty, self.request_topic, self._on_request, 10)
        self._service = node.create_service(Trigger, self.service_name, self._on_service)

        # Only this timer touches ROS; scanning happens off the spin thread so a
        # rescan never stalls the render loop.
        self._timer = node.create_timer(PUBLISH_TICK_S, self._tick)

        self._scan_thread = threading.Thread(
            target=self._scan_loop, name="avatar-list-scan", daemon=True)
        self._scan_thread.start()

        self.logger.info(
            f"[avatar-list] serving {self.service_name} "
            f"(reply: {self.reply_topic}, request: {self.request_topic})")

    # -- public API -------------------------------------------------------- #

    def resolve(self, name: Optional[str]):
        """Map a string from /operator/name to an :class:`Avatar`, or None.

        Safe to call from the ROS thread: the lookup table is swapped as a
        whole by the scan thread.
        """
        return self.registry.resolve(name)

    def id_for_target(self, target: str) -> str:
        """Id the app uses for an avatar folder (or "ai"), "" if unlisted."""
        return self.registry.id_for_target(target)

    def set_active(self, avatar_id: Optional[str]) -> None:
        """Tell the app which avatar is being rendered right now.

        Applied by the scan thread and published within a tick.  ``revision``
        does not change, so the app only moves its selection.
        """
        with self._lock:
            self._desired_active = avatar_id or ""
        self._wake.set()

    def request_rescan(self) -> None:
        """Force a rescan now instead of waiting for the next poll."""
        self._wake.set()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()

    # -- ROS callbacks ----------------------------------------------------- #

    def _on_service(self, _request, response):
        with self._lock:
            ready = self._ready
        payload = self.registry.payload
        if ready and payload:
            response.success = True
            response.message = payload
        else:
            response.success = False
            response.message = "avatar list not ready yet: the first scan of the avatar directory has not completed"
        return response

    def _on_request(self, _msg):
        with self._lock:
            ready = self._ready
        payload = self.registry.payload
        if ready and payload:
            self._publish(payload)
        else:
            self.logger.warning(
                "[avatar-list] request received before the first scan finished; "
                "will publish as soon as the list is ready")
            with self._lock:
                self._publish_pending = True

    def _tick(self):
        with self._lock:
            if not (self._ready and self._publish_pending):
                return
            self._publish_pending = False
        payload = self.registry.payload
        revision, active, count = (self.registry.revision,
                                   self.registry.active,
                                   len(self.registry.avatars))
        self._publish(payload)
        self.logger.info(
            f"[avatar-list] published {count} avatar(s), revision {revision}, active '{active}'")

    def _publish(self, payload: str) -> None:
        msg = String()
        msg.data = payload
        self._reply_pub.publish(msg)

    # -- scanning thread --------------------------------------------------- #

    def _scan_loop(self):
        signature = None
        last_full_scan = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            changed = False
            try:
                current = self.registry.directory_signature()
                due = (now - last_full_scan) >= self.rescan_interval_s
                if current != signature or due or not self._ready:
                    signature = current
                    last_full_scan = now
                    changed = self.registry.scan()

                # Applied after the scan as well: an id selected before the
                # first scan only becomes resolvable once the list exists.
                with self._lock:
                    desired_active = self._desired_active
                changed |= self.registry.set_active(desired_active)

                with self._lock:
                    if changed or not self._ready:
                        self._publish_pending = True
                    self._ready = True
            except Exception as exc:  # never let the scan thread die
                self.logger.error(f"[avatar-list] scan failed: {exc}")

            self._wake.wait(self.poll_interval_s)
            self._wake.clear()


# --------------------------------------------------------------------------- #
# Standalone node: python3 ros2_avatar_list.py [--robot NAME] [--root DIR]
#
# The avatar list normally lives inside the viewer's node (see
# ros2_blendshape_node.py). This entry point runs it on its own, which is handy
# for testing the service and the topics without starting the renderer. It has
# no renderer, so avatars without a thumbnail.jpg get the placeholder tile.
# --------------------------------------------------------------------------- #

def main(argv=None):
    import argparse

    import rclpy
    from rclpy.node import Node

    parser = argparse.ArgumentParser(description="Serve the avatar list on its own node.")
    parser.add_argument("--robot", default=None, help="robot name used in the topic namespace")
    parser.add_argument("--root", default=DEFAULT_AVATAR_ROOT, help="avatar directory")
    args, ros_args = parser.parse_known_args(argv)

    try:
        from ros2_blendshape_node import NAME_MAP
    except Exception:
        NAME_MAP = None

    rclpy.init(args=ros_args)
    node = Node("avatar_list")
    interface = AvatarListInterface(
        node=node, avatar_root=args.root, legacy_name_map=NAME_MAP, robot_name=args.robot)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        interface.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
