import atexit
import threading
import time

import numpy as np

from .. import logger
from ..spatial_types.spatial_types import RotationMatrix

try:
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import Vector3, Point, Quaternion, Pose
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Vector3, Point, PoseStamped, Quaternion, Pose
except ImportError as e:
    logger.warning(
        f"Could not import ros messages, viz marker will not be available: {e}"
    )

from ..world_description.geometry import (
    FileMesh,
    Box,
    Sphere,
    Cylinder,
    Primitive,
    TriangleMesh,
)
from ..world import World


class VizMarkerPublisher:
    """
    Publishes an Array of visualization marker which represent the situation in the World
    """

    def __init__(
        self,
        world: World,
        node,
        topic_name="/semworld/viz_marker",
        interval=0.1,
        reference_frame="map",
        visuals_if_available=False,
    ):
        """
        The Publisher creates an Array of Visualization marker with a Marker for each body in the World and publishes
        it to the given topic name at a fixed interval. The publisher automatically stops publishing when the process
        is killed.

        :param world: The World to which the Visualization Marker should be published.
        :param node: The ROS2 node that will be used to publish the visualization marker.
        :param topic_name: The name of the topic to which the Visualization Marker should be published.
        :param interval: The interval at which the visualization marker should be published, in seconds.
        :param reference_frame: The reference frame of the visualization marker.
        """
        self.interval = interval
        self.reference_frame = reference_frame
        self.world = world
        self.node = node
        self.visuals_if_available = visuals_if_available

        self.pub = self.node.create_publisher(MarkerArray, topic_name, 10)

        self.world_calback = lambda: self._publish()
        self.world.state_change_callbacks.append(self.world_calback)

    def _publish(self) -> None:
        """
        Constantly publishes the Marker Array. To the given topic name at a fixed rate.
        """
        marker_array = self._make_marker_array()
        self.pub.publish(marker_array)

    def _make_marker_array(self) -> MarkerArray:
        """
        Creates the Marker Array to be published. There is one Marker for link for each object in the Array, each Object
        creates a name space in the visualization Marker. The type of Visualization Marker is decided by the collision
        tag of the URDF.

        :return: An Array of Visualization Marker
        """
        marker_array = MarkerArray()
        for body in self.world.bodies:
            shapes = (
                body.visual
                if self.visuals_if_available and body.visual and len(body.visual) > 0
                else body.collision
            )
            for i, shape in enumerate(shapes):
                msg = Marker()
                msg.header.frame_id = self.reference_frame
                msg.ns = body.name.name
                msg.id = i
                msg.action = Marker.ADD
                msg.pose = self.transform_to_pose(
                    (
                        self.world.compute_forward_kinematics(self.world.root, body)
                        @ shape.origin
                    ).to_np()
                )
                msg.color = (
                    ColorRGBA(
                        r=float(shape.color.R),
                        g=float(shape.color.G),
                        b=float(shape.color.B),
                        a=float(shape.color.A),
                    )
                    if isinstance(shape, Primitive)
                    else ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                )
                msg.lifetime = Duration(sec=100)

                if isinstance(shape, FileMesh):
                    msg.type = Marker.MESH_RESOURCE
                    msg.mesh_resource = "file://" + shape.filename
                    msg.scale = Vector3(
                        x=float(shape.scale.x),
                        y=float(shape.scale.y),
                        z=float(shape.scale.z),
                    )
                    msg.mesh_use_embedded_materials = True
                elif isinstance(shape, TriangleMesh):
                    f = shape.file
                    msg.type = Marker.MESH_RESOURCE
                    msg.mesh_resource = "file://" + f.name
                    msg.scale = Vector3(
                        x=float(shape.scale.x),
                        y=float(shape.scale.y),
                        z=float(shape.scale.z),
                    )
                    msg.mesh_use_embedded_materials = True
                elif isinstance(shape, Cylinder):
                    msg.type = Marker.CYLINDER
                    msg.scale = Vector3(
                        x=float(shape.width),
                        y=float(shape.width),
                        z=float(shape.height),
                    )
                elif isinstance(shape, Box):
                    msg.type = Marker.CUBE
                    msg.scale = Vector3(
                        x=float(shape.scale.x),
                        y=float(shape.scale.y),
                        z=float(shape.scale.z),
                    )
                elif isinstance(shape, Sphere):
                    msg.type = Marker.SPHERE
                    msg.scale = Vector3(
                        x=float(shape.radius * 2),
                        y=float(shape.radius * 2),
                        z=float(shape.radius * 2),
                    )

                marker_array.markers.append(msg)
        return marker_array

    def _stop_publishing(self) -> None:
        """
        Stops the publishing of the Visualization Marker update by setting the kill event and collecting the thread.
        """
        self.world.state_change_callbacks.remove(self.world_calback)

    def __del__(self) -> None:
        try:
            self._stop_publishing()
        except Exception:
            pass
        finally:
            super().__del__()

    @staticmethod
    def transform_to_pose(transform: np.ndarray) -> Pose:
        """
        Converts a 4x4 transformation matrix to a PoseStamped message.

        :param transform: The transformation matrix to convert.
        :return: A PoseStamped message.
        """
        pose = Pose()
        pose.position = Point(**dict(zip(["x", "y", "z"], transform[:3, 3])))
        pose.orientation = Quaternion(
            **dict(
                zip(
                    ["x", "y", "z", "w"],
                    RotationMatrix(transform).to_quaternion().to_np(),
                )
            )
        )
        return pose
