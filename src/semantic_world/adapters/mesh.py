import logging
import os
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import trimesh

from semantic_world.connections import Connection6DoF, FixedConnection
from semantic_world.spatial_types.spatial_types import RotationMatrix
from ..geometry import Mesh, TriangleMesh, Scale
from ..prefixed_name import PrefixedName
from ..spatial_types.spatial_types import TransformationMatrix, Point3
from ..world import World
from ..world_entity import Body
import fbxloader
from fbxloader.nodes import Mesh as FBXMesh, Object3D, Scene

@dataclass
class MeshParser:
    """
    Adapter for mesh files.
    """

    file_path: str
    """
    The path to the mesh file.
    """
    scale: Scale = field(default_factory=Scale)
    """
    The scale to apply to the mesh.
    """

    def parse(self) -> World:
        """
        Parse the mesh file to a body and return a world containing that body.

        :return: A World object containing the parsed body.
        """
        file_name = os.path.basename(self.file_path)

        mesh_shape = Mesh(
            origin=TransformationMatrix(), filename=self.file_path, scale=self.scale
        )
        body = Body(
            name=PrefixedName(file_name), collision=[mesh_shape], visual=[mesh_shape]
        )

        world = World()
        world.add_body(body)

        return world


@dataclass
class STLParser(MeshParser):
    pass


@dataclass
class OBJParser(MeshParser):
    pass


@dataclass
class DAEParser(MeshParser):
    pass


@dataclass
class PLYParser(MeshParser):
    pass


@dataclass
class OFFParser(MeshParser):
    pass


@dataclass
class GLBParser(MeshParser):
    pass


@dataclass
class XYZParser(MeshParser):
    pass


@dataclass
class CoordinateAxis(Enum):

    X = (0, 1)
    NEGATIVE_X = (0, -1)
    Y = (1, 1)
    NEGATIVE_Y = (1, -1)
    Z = (2, 1)
    NEGATIVE_Z = (2, -1)

    @classmethod
    def from_fbx(cls, axis_value: int, sign: int):
        """Map FBX axis index + sign into a CoordinateAxis enum."""
        for member in cls:
            if member.value[0] == axis_value and member.value[1] == sign:
                return member

    def to_vector(self):
        idx, sgn = self.value
        v = np.zeros(3)
        v[idx] = float(sgn)
        return v

@dataclass
class FBXGlobalSettings:

    fbx_loader: fbxloader.FBXLoader

    up_axis: CoordinateAxis = field(init=False)
    front_axis: CoordinateAxis = field(init=False)
    coord_axis: CoordinateAxis = field(init=False)

    def __post_init__(self):
        fbx = self.fbx_loader

        self.up_axis = CoordinateAxis.from_fbx(
            fbx.fbxtree["GlobalSettings"]["UpAxis"]["value"],
            fbx.fbxtree["GlobalSettings"]["UpAxisSign"]["value"],
        )

        self.front_axis = CoordinateAxis.from_fbx(
            fbx.fbxtree["GlobalSettings"]["FrontAxis"]["value"],
            fbx.fbxtree["GlobalSettings"]["FrontAxisSign"]["value"],
        )

        self.coord_axis = CoordinateAxis.from_fbx(
            fbx.fbxtree["GlobalSettings"]["CoordAxis"]["value"],
            fbx.fbxtree["GlobalSettings"]["CoordAxisSign"]["value"],
        )

    def get_fbx_T_semantic_world(self):
        """
        Get the transformation matrix from FBX to Semantic World coordinate system.
        """
        sX = self.front_axis.to_vector()
        sY = self.coord_axis.to_vector()
        sZ = self.up_axis.to_vector()
        S = np.column_stack((sX, sY, sZ))

        R = S.T

        M = np.eye(4)
        M[:3, :3] = R
        return M


@dataclass
class FBXParser(MeshParser):
    """
    Adapter for FBX files.
    """

    @staticmethod
    def transform_vertices(vertices: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        vertices: (N, 3) float array
        M: (4, 4) transform that maps FBX -> Semantic World (column-vector convention)
        returns transformed vertices (N, 3)
        """
        assert vertices.ndim == 2 and vertices.shape[1] == 3, "vertices must be (N,3)"
        assert M.shape == (4, 4), "M must be 4x4"

        # make homogeneous row vectors
        ones = np.ones((vertices.shape[0], 1), dtype=vertices.dtype)
        Vh = np.concatenate([vertices, ones], axis=1)  # (N,4)

        # row-vector transform ⇒ right-multiply by M^T
        Vh_out = Vh @ M.T  # (N,4)
        return Vh_out[:, :3]

    def parse(self) -> World:
        """
        Parse the FBX file, each object in the FBX file is converted to a body in the world and the meshes are loaded
        as TriangleMesh objects.

        :return: A World containing content of the FBX file.
        """
        fbx = fbxloader.FBXLoader(self.file_path)

        global_settings = FBXGlobalSettings(fbx)
        fbx_T_semantic_world = global_settings.get_fbx_T_semantic_world()

        world = World()

        id_to_centroid_map = {}

        with world.modify_world():
            for obj_id, obj in fbx.objects.items():
                # Create a body for each object in the FBX file
                if type(obj) is Object3D:
                    name = fbx.fbxtree["Objects"]["Model"][obj_id]["attrName"].split(
                        "\x00"
                    )[0]
                    meshes = []
                    for o in obj.children:
                        if isinstance(o, FBXMesh):
                            aligned_vertices = self.transform_vertices(o.vertices, fbx_T_semantic_world) / 100

                            # center = aligned_vertices.mean(axis=0)  # centroid

                            #
                            # assert not obj in id_to_centroid_map, f"Object ID {obj_id} already has a center assigned, we now need to handle cases with multiple meshes per object ID."
                            # id_to_centroid_map[obj_id] = center
                            #
                            # # Shift vertices
                            # aligned_vertices = aligned_vertices - center
                            #
                            # # Update object transform: pre-translate by +center
                            # T = np.eye(4)
                            # T[:3, 3] = center

                            t_mesh = TriangleMesh(origin=TransformationMatrix(), mesh=trimesh.Trimesh(vertices=aligned_vertices,faces=o.faces))

                            meshes.append(t_mesh)
                    body = Body(
                        name=PrefixedName(name), collision=meshes, visual=meshes
                    )
                    world.add_body(body)

            for obj in fbx.objects.values():
                if type(obj) is Object3D:
                    name = fbx.fbxtree["Objects"]["Model"][obj.id]["attrName"].split(
                        "\x00"
                    )[0]
                    parent_name = (
                        fbx.fbxtree["Objects"]["Model"][obj.parent.id][
                            "attrName"
                        ].split("\x00")[0]
                        if type(obj.parent) is not Scene
                        else None
                    )
                    if not parent_name:
                        continue

                    obj_body = world.get_body_by_name(name)
                    parent_body = world.get_body_by_name(parent_name)

                    # world_P_obj = id_to_centroid_map[obj.id]
                    # world_P_parent = id_to_centroid_map.get(obj.parent.id, np.zeros(3))
                    # parent_P_obj = world_P_obj - world_P_parent
                    #
                    # translation = Point3(*parent_P_obj)

                    translation = Point3(*obj.matrix[3, :3])
                    rotation_matrix = RotationMatrix(obj.matrix)
                    parent_T_child = TransformationMatrix.from_point_rotation_matrix(
                        translation, rotation_matrix, reference_frame=parent_body
                    )

                    connection = FixedConnection(
                        parent=parent_body,
                        child=obj_body,
                        _world=world,
                        origin_expression=parent_T_child,
                    )
                    world.add_connection(connection)

        return world
