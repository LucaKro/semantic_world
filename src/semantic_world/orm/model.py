from dataclasses import dataclass, field
from typing import List, Optional

import trimesh
from io import BytesIO

from ormatic.dao import AlternativeMapping, T

from ..degree_of_freedom import DegreeOfFreedom
from ..prefixed_name import PrefixedName
from ..spatial_types import RotationMatrix, Vector3, Point3, TransformationMatrix
from ..spatial_types.derivatives import DerivativeMap
from ..spatial_types.spatial_types import Quaternion
from ..spatial_types.symbol_manager import symbol_manager
from ..world import World, Body
from ..world_entity import Connection, View, WorldEntity
from typing import Type, Optional
from sqlalchemy import TypeDecorator
from sqlalchemy import types
import trimesh.exchange.stl


@dataclass
class WorldMapping(AlternativeMapping[World]):
    root: Body
    bodies: List[Body]
    connections: List[Connection]
    views: List[View]
    degrees_of_freedom: List[DegreeOfFreedom]

    @classmethod
    def create_instance(cls, obj: World):
        return cls(obj.root, obj.bodies, obj.connections, obj.views, list(obj.degrees_of_freedom))

    def create_from_dao(self) -> World:
        result = World()

        with result.modify_world():
            for body in self.bodies:
                result.add_body(body)
            for connection in self.connections:
                result.add_connection(connection)
            for view in self.views:
                result.add_view(view)
            for dof in self.degrees_of_freedom:
                result.create_degree_of_freedom(name=dof.name, lower_limits=dof.lower_limits,
                                                upper_limits=dof.upper_limits)

        return result


@dataclass
class Vector3Mapping(AlternativeMapping[Vector3]):
    x: float
    y: float
    z: float

    reference_frame: Optional[Body] = field(init=False, default=None)

    @classmethod
    def create_instance(cls, obj: Vector3):
        x, y, z, _ = symbol_manager.evaluate_expr(obj).tolist()
        result = cls(x=x, y=y, z=z)
        result.reference_frame = obj.reference_frame
        return result

    def create_from_dao(self) -> Vector3:
        return Vector3(x=self.x, y=self.y, z=self.z, reference_frame=self.reference_frame)


@dataclass
class Point3Mapping(AlternativeMapping[Point3]):
    x: float
    y: float
    z: float

    reference_frame: Optional[Body] = field(init=False, default=None)

    @classmethod
    def create_instance(cls, obj: Point3):
        x, y, z, _ = symbol_manager.evaluate_expr(obj).tolist()
        result = cls(x=x, y=y, z=z)
        result.reference_frame = obj.reference_frame
        return result

    def create_from_dao(self) -> Point3:
        return Point3(x=self.x, y=self.y, z=self.z, reference_frame=self.reference_frame)


@dataclass
class QuaternionMapping(AlternativeMapping[Quaternion]):
    x: float
    y: float
    z: float
    w: float

    reference_frame: Optional[Body] = field(init=False, default=None)

    @classmethod
    def create_instance(cls, obj: Quaternion):
        x, y, z, w = symbol_manager.evaluate_expr(obj).tolist()
        result = cls(x=x, y=y, z=z, w=w)
        result.reference_frame = obj.reference_frame
        return result

    def create_from_dao(self) -> Quaternion:
        return Quaternion(x=self.x, y=self.y, z=self.z, w=self.w, reference_frame=self.reference_frame)


@dataclass
class RotationMatrixMapping(AlternativeMapping[RotationMatrix]):
    rotation: Quaternion
    reference_frame: Optional[Body] = field(init=False, default=None)

    @classmethod
    def create_instance(cls, obj: RotationMatrix):
        result = cls(rotation=obj.to_quaternion())
        result.reference_frame = obj.reference_frame
        return result

    def create_from_dao(self) -> RotationMatrix:
        result = RotationMatrix.from_quaternion(self.rotation)
        result.reference_frame = self.reference_frame
        return result


@dataclass
class TransformationMatrixMapping(AlternativeMapping[TransformationMatrix]):
    position: Point3
    rotation: Quaternion
    reference_frame: Optional[Body] = field(init=False, default=None)
    child_frame: Optional[Body] = field(init=False, default=None)

    @classmethod
    def create_instance(cls, obj: TransformationMatrix):
        position = obj.to_position()
        rotation = obj.to_quaternion()
        result = cls(position=position,
                     rotation=rotation)
        result.reference_frame = obj.reference_frame
        result.child_frame = obj.child_frame

        return result

    def create_from_dao(self) -> TransformationMatrix:
        result = TransformationMatrix.from_point_rotation_matrix(
            point=self.position,
            rotation_matrix=RotationMatrix.from_quaternion(self.rotation), reference_frame=self.reference_frame,
            child_frame=self.child_frame, )
        return result

@dataclass
class DegreeOfFreedomMapping(WorldEntity, AlternativeMapping[DegreeOfFreedom]):
    lower_limits: List[float]
    upper_limits: List[float]

    @classmethod
    def create_instance(cls, obj: DegreeOfFreedom):
        return cls(name=obj.name, lower_limits=obj.lower_limits.data, upper_limits=obj.upper_limits.data)

    def create_from_dao(self) -> DegreeOfFreedom:
        lower_limits = DerivativeMap(data=self.lower_limits)
        upper_limits = DerivativeMap(data=self.upper_limits)
        return DegreeOfFreedom(name=self.name, lower_limits=lower_limits, upper_limits=upper_limits)


class TrimeshType(TypeDecorator):
    """
    Type that casts fields that are of type `type` to their class name on serialization and converts the name
    to the class itself through the globals on load.
    """
    impl = types.LargeBinary(4 * 1024 * 1024 * 1024 - 1) # 4 GB max

    def process_bind_param(self, value: trimesh.Trimesh, dialect):
        # return binary version of trimesh
        return trimesh.exchange.stl.export_stl(value)

    def process_result_value(self, value: impl, dialect) -> Optional[trimesh.Trimesh]:
        if value is None:
            return None
        mesh = trimesh.Trimesh(**trimesh.exchange.stl.load_stl_binary(BytesIO(value)))
        return mesh
