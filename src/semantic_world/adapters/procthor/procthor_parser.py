import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import rclpy
from ormatic.utils import drop_database
from sqlalchemy import create_engine
from sqlalchemy import select, exists, and_, or_
from sqlalchemy.orm import Session

from semantic_world.adapters.viz_marker import VizMarkerPublisher
from semantic_world.connections import FixedConnection
from semantic_world.geometry import Scale
from semantic_world.orm.ormatic_interface import (
    WorldMappingDAO,
    BodyDAO,
    ViewDAO,
    ConnectionDAO,
    PrefixedNameDAO,
    Base,
)
from semantic_world.prefixed_name import PrefixedName
from semantic_world.spatial_types.spatial_types import (
    TransformationMatrix,
    Point3,
    RotationMatrix,
)
from semantic_world.variables import SpatialVariables
from semantic_world.views.factories import (
    DoorFactory,
    RoomFactory,
    WallFactory,
    HandleFactory,
    Direction,
)
from semantic_world.world import World
from semantic_world.world_entity import Body, Region


@dataclass
class ProcthorWall:
    """
    Procthor wall, represented by two polygon walls perfectly overlapping, and a list of dictionaries
    representing door holes which may be empty
    """

    walls: List[dict] = field(default_factory=list)
    """
    List of dictionaries, where each dictionary represents one wall polygon in procthor
    """

    doors: List[dict] = field(default_factory=list)
    """
    List of dictionaries, where each dictionary represents one door hole in the wall polygon
    """


def unity4x4_to_sdt4x4(unity_transform_matrix: np.ndarray):
    """
    Convert a left-handed Y-up, Z-forward Unity transform to the right-handed Z-up, X-forward convention used in the
    semantic digital twin.

    :param unity_transform_matrix: (4, 4) shaped np.ndarray since we need to use reflection
    returns: (4,4) numpy.ndarray representing the input transform, in the fixed coordinate convention
    """

    assert unity_transform_matrix.shape == (
        4,
        4,
    ), "unity_transform_matrix is not a 4x4 shaped np.ndarray."

    permutation_matrix = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=float,
    )

    reflection_vector = np.diag([1, -1, 1])
    R = reflection_vector @ permutation_matrix
    conjugation_matrix = np.eye(4)
    conjugation_matrix[:3, :3] = R
    inverse_conjugation_matrix = conjugation_matrix.T

    unity_transform_matrix = np.asarray(unity_transform_matrix, float).reshape(4, 4)

    return conjugation_matrix @ unity_transform_matrix @ inverse_conjugation_matrix


def process_door_polygon(
    door: dict, wall_width: float, door_thickness: float = 0.02
) -> Tuple[Scale, Point3]:
    """
    Extracts the door scale from the doorÄs 'holePolygon', and uses wall width to compute the doors translation
    with the wall as its reference frame

    :param door: dictionary of the door
    :param wall_width: width of the wall
    :param door_thickness: thickness of the door

    :returns: Scale representing the doors geometry and Point3 representing the doors position from the walls perspective
    """

    door_polygon = door["holePolygon"]
    x0, y0 = float(door_polygon[0]["x"]), float(door_polygon[0]["y"])
    x1, y1 = float(door_polygon[1]["x"]), float(door_polygon[1]["y"])

    xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
    ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)

    door_width = xmax - xmin
    door_height = ymax - ymin
    door_scale = Scale(door_width, door_height, door_thickness)

    # Door center origin expressed from the wall's horizontal center. Unity's wall origin is in one of the corners
    x_origin_wall_corner = 0.5 * (xmin + xmax)
    y_origin = 0.5 * (ymin + ymax)
    x_origin = x_origin_wall_corner - 0.5 * wall_width

    return door_scale, Point3(x_origin, y_origin, 0.0)


@dataclass
class ProcTHORParser:
    file_path: str
    session: Session

    @staticmethod
    def import_room(room: dict) -> Tuple[World, TransformationMatrix]:
        """
        Processes a room dictionary by creating a region from the 'floorPolygon'.

        :param room: room dictionary

        :returns: Room World and Transformation Matrix
        """
        room_id = room["id"].split("|")[-1]
        room_name = PrefixedName(f"{room["roomType"]}_{room_id}")

        # converting unity coordinates to semantic digital twin coordinates
        room_polytope: List[Point3] = [
            Point3(v["z"], -v["x"], v["y"]) for v in room["floorPolygon"]
        ]

        polytope_length = len(room_polytope)
        x_center = sum(v.x.to_np() for v in room_polytope) / polytope_length
        y_center = sum(v.y.to_np() for v in room_polytope) / polytope_length
        z_center = sum(v.z.to_np() for v in room_polytope) / polytope_length

        center_point = Point3(x_center, y_center, z_center)

        centered_polytope = [
            Point3(
                v.x.to_np() - x_center, v.y.to_np() - y_center, v.z.to_np() - z_center
            )
            for v in room_polytope
        ]

        region = Region.from_3d_points(
            points_3d=centered_polytope,
            drop_dimension=SpatialVariables.z,
            name=PrefixedName(room_name.name + "_region"),
            reference_frame=Body(name=room_name),
        )

        room_factory = RoomFactory(name=room_name, region=region)

        transform = TransformationMatrix.from_point_rotation_matrix(center_point)

        return room_factory.create(), transform

    def import_object(self, obj: dict) -> Tuple[World, TransformationMatrix]:
        """
        Processes an object dictionary by querying its world from the database using its 'asset_id', and recursively
        importing its children. If the object is not found inside the database, a virtual body is created and the
        children are still processed

        :param obj: object dictionary

        :returns: Object World and Transformation Matrix
        """
        asset_id = obj["assetId"]

        body_world: World = get_world_by_prefixed_name(self.session, name=asset_id)

        if body_world is None:
            logging.error(
                f"Could not find asset {asset_id} in the database. Using virtual body and proceeding to process children"
            )
            body_world = World(name=asset_id)
            body_world_root = Body(name=PrefixedName(asset_id))
            body_world.add_body(body_world_root)

        body_transform = TransformationMatrix.from_xyz_rpy(
            obj["position"]["x"],
            obj["position"]["y"],
            obj["position"]["z"],
            obj["rotation"]["x"],
            obj["rotation"]["y"],
            obj["rotation"]["z"],
        ).to_np()

        body_transform = TransformationMatrix(unity4x4_to_sdt4x4(body_transform))

        for child in obj.get("children", {}):
            child_world, child_transform = self.import_object(child)
            child_connection = FixedConnection(
                parent=body_world.root,
                child=child_world.root,
                origin_expression=child_transform,
            )
            body_world.merge_world(child_world, child_connection)

        return body_world, body_transform

    @staticmethod
    def group_walls_by_polygon(
        walls_without_doors: List[Dict],
    ) -> List[ProcthorWall]:
        """
        Groups walls with identical polygons (order-invariant) into pairs -> ProcthorWall(walls=[w1, w2]).
        Asserts that all doors input can be paired without any leftovers, since each physical wall in procthor
        consists of two polygons.

        :param walls_without_doors: List of walls without doors

        :return: List of ProcthorWall
        """

        def polygon_key(poly):
            return frozenset((p["x"], p["y"], p["z"]) for p in poly)

        groups: Dict[frozenset, List[Dict]] = {}
        for wall in walls_without_doors:
            key = polygon_key(wall.get("polygon", []))
            groups.setdefault(key, []).append(wall)

        paired_walls: List[ProcthorWall] = []
        for walls in groups.values():
            i = 0
            while i + 1 < len(walls):
                paired_walls.append(ProcthorWall(walls=[walls[i], walls[i + 1]]))
                i += 2
            if i < len(walls):
                raise AssertionError(
                    "There are cases were a physical wall is represented by only one polygon. This case may need to be handled now"
                )

        return paired_walls

    def process_door_wall_combinations(
        self, doors: List[Dict], walls: List[Dict]
    ) -> List[ProcthorWall]:
        """
        Processes grouped walls and doors into one ProcthorWall. This is done by first looking for wall references in
        the doors 'wall0' and 'wall1'. the remaining walls are grouped by matching polygons.

        :param doors: List of door dictionaries
        :param walls: List of wall dictionaries

        :returns: List of ProcthorWall
        """

        walls_by_id = {wall["id"]: wall for wall in walls}
        used_wall_ids = set()
        door_groups = []

        for door in doors:
            ids = [door.get("wall0"), door.get("wall1")]
            found = []
            for wid in ids:
                if not wid:
                    continue
                w = walls_by_id.get(wid)
                if w:
                    found.append(w)
                    used_wall_ids.add(wid)
            door_groups.append(ProcthorWall(doors=[door], walls=found))

        remaining_walls = [w for w in walls if w["id"] not in used_wall_ids]

        paired_walls = self.group_walls_by_polygon(remaining_walls)

        door_groups.extend(paired_walls)

        return door_groups

    @staticmethod
    def process_wall_polygon(
        polygon: List[Dict], wall_thickness: float = 0.02
    ) -> Tuple[Scale, TransformationMatrix]:
        """
        Constructs wall scale and transform from wall polygon, with its origins height on y=0
        Transform is computed using the centerpoint of the wall polygon, and calculating the yaw using atan2(dz, -dx)

        :param polygon: List of wall polygon dictionaries
        :param wall_thickness: Thickness of wall polygon, defaults to 0.02

        :returns: Scale, TransformationMatrix
        """

        xz_sets = {}
        for p in polygon:
            key = (float(p["x"]), float(p["z"]))
            xz_sets.setdefault(key, []).append(float(p["y"]))

        if len(xz_sets) != 2:
            raise ValueError("Expected exactly two unique (x,z) positions.")

        (x1, z1), y_coords1 = list(xz_sets.items())[0]
        (x2, z2), y_coords2 = list(xz_sets.items())[1]

        # Height from one vertical pair
        min_y, max_y = min(min(y_coords1, y_coords2)), max(max(y_coords1, y_coords2))
        height = max_y - min_y

        # Horizontal length
        dx, dz = x2 - x1, z2 - z1
        width = math.hypot(dx, dz)

        # Center
        x_center = (x1 + x2) * 0.5
        z_center = (z1 + z2) * 0.5

        yaw = math.atan2(dz, -dx)

        scale = Scale(x=width, y=height, z=wall_thickness)
        transform = TransformationMatrix.from_xyz_rpy(
            x_center, 0, z_center, 0.0, yaw, 0
        )
        return scale, transform

    def import_walls(
        self, procthor_wall: ProcthorWall
    ) -> Tuple[World, TransformationMatrix]:
        """
        Takes a ProcthorWall object and processes it. The wall polygon is used to create a box of that size, and
         if ProcthorWall also has a door, a hole in the shape of a that door is cut out of the wall. Then a door
         is created and placed at the correct position as the walls child.

        :param procthor_wall: ProcthorWall object

        :return: World, TransformationMatrix
        """

        # In unity there are always two wall polygons, one for each side of the physical wall. these walls are both
        # referenced inside the wall dictionary as "wall0", and "wall1". The hole polygon of the door is defined
        # from the corner origin of the wall0 polygon, so we need to use that wall. If we use the other one, hole is
        # on the wrong side of the wall.
        if procthor_wall.doors:
            used_wall = (
                procthor_wall.walls[0]
                if procthor_wall.walls[0]["roomId"] == procthor_wall.doors[0]["wall0"]
                else procthor_wall.walls[1]
            )
        else:
            used_wall = procthor_wall.walls[0]

        room_numbers = [w["id"].split("|")[1] for w in procthor_wall.walls]
        wall_corners = used_wall["id"].split("|")[2:]
        wall_name = PrefixedName(
            f"wall_{wall_corners[0]}_{wall_corners[1]}_{wall_corners[2]}_{wall_corners[3]}_room{room_numbers[0]}_room{room_numbers[1]}"
        )

        wall_scale, wall_transform = self.process_wall_polygon(used_wall["polygon"])

        # Scale cannot be negative, so the usual minus in front of wall_scale.x omitted
        wall_scale = Scale(wall_scale.z, wall_scale.x, wall_scale.y)

        wall_transform = unity4x4_to_sdt4x4(wall_transform.to_np())

        # The wall is artificially set to z=0 here, because
        # 1. as of now, procthor house floors have the same z value
        # 2. Since doors origins are in 3d center, positioning the door correctly at the floor given potentially varying
        #    wall heights is unnecessarily complex given the assumption stated in 1.
        wall_transform = TransformationMatrix.from_point_rotation_matrix(
            Point3(*wall_transform[:3, 3]),
            RotationMatrix(wall_transform),
        )

        door_factories = []
        door_transforms = []

        for door in procthor_wall.doors:
            room_numbers = door["id"].split("|")[1:]

            door_name = PrefixedName(
                f"{door["assetId"]}_room{room_numbers[0]}_room{room_numbers[1]}"
            )
            handle_name = PrefixedName(
                f"{door["assetId"]}_room{room_numbers[0]}_room{room_numbers[1]}_handle"
            )

            # In unity, doors are defined as holes in the wall, so we express them as children of walls.
            # This means we just need to translate them, and can assume no rotation
            door_scale, door_position = process_door_polygon(
                door, wall_width=wall_scale.y
            )

            door_scale = Scale(door_scale.z, door_scale.x, door_scale.y)
            door_position = Point3(
                door_position.z.to_np(),
                -door_position.x.to_np(),
                door_position.y.to_np(),
            )

            door_transform = TransformationMatrix.from_point_rotation_matrix(
                door_position,
            )

            # I think a double door factory makes sense here, since it allows us to make assumptions about joints, scales, positions etc here
            door_factory = DoorFactory(
                name=door_name,
                scale=door_scale,
                handle_factory=HandleFactory(name=handle_name),
                handle_direction=Direction.Y,
            )

            door_factories.append(door_factory)
            door_transforms.append(door_transform)

        wall_factory = WallFactory(
            name=wall_name,
            scale=wall_scale,
            door_factories=door_factories,
            door_transforms=door_transforms,
        )

        return wall_factory.create(), wall_transform

    def parse(self) -> World:
        """
        Parses a JSON file from procthor into a world.
        Rooms are represented as Regions
        Walls and doors are constructed from the supplied polygons
        Objects are imported from the database
        """
        with open(self.file_path) as f:
            house = json.load(f)
        house_name = self.file_path.split("/")[-1].split(".")[0]

        world = World(name=house_name)
        world_root = Body(name=PrefixedName(house_name))
        world.add_body(world_root)

        for room in house["rooms"]:
            room_world, room_transform = self.import_room(room)
            room_connection = FixedConnection(
                parent=world.root,
                child=room_world.root,
                origin_expression=room_transform,
            )
            world.merge_world(room_world, room_connection)

        for obj in house["objects"]:
            obj_world, obj_transform = self.import_object(obj)
            obj_connection = FixedConnection(
                parent=world.root, child=obj_world.root, origin_expression=obj_transform
            )
            world.merge_world(obj_world, obj_connection)

        doors = house["doors"]
        walls = house["walls"]

        procthor_walls: List[ProcthorWall] = self.process_door_wall_combinations(
            doors, walls
        )

        for procthor_wall in procthor_walls:
            wall_world, wall_transform = self.import_walls(procthor_wall)
            wall_connection = FixedConnection(
                parent=world.root,
                child=wall_world.root,
                origin_expression=wall_transform,
            )
            world.merge_world(wall_world, wall_connection)

        return world


def get_world_by_prefixed_name(
    session: Session, name: str, prefix: Optional[str] = None
) -> Optional[World]:
    """
    To be deleted as soon as ORM fully integrated the EQL interface
    """

    # Helper to build the name filter for PrefixedNameDAO
    def pn_filter():
        base = PrefixedNameDAO.name == name
        return (
            and_(base, PrefixedNameDAO.prefix == prefix) if prefix is not None else base
        )

    # EXISTS subqueries for bodies, views, and connections
    bodies_exist = exists(
        select(BodyDAO.id)
        .join(PrefixedNameDAO, BodyDAO.name)  # WorldEntity.name relationship
        .where(BodyDAO.worldmappingdao_bodies_id == WorldMappingDAO.id, pn_filter())
        .limit(1)
    )

    views_exist = exists(
        select(ViewDAO.id)
        .join(PrefixedNameDAO, ViewDAO.name)
        .where(ViewDAO.worldmappingdao_views_id == WorldMappingDAO.id, pn_filter())
        .limit(1)
    )

    conns_exist = exists(
        select(ConnectionDAO.id)
        .join(PrefixedNameDAO, ConnectionDAO.name)
        .where(
            ConnectionDAO.worldmappingdao_connections_id == WorldMappingDAO.id,
            pn_filter(),
        )
        .limit(1)
    )

    q = select(WorldMappingDAO).where(or_(bodies_exist, views_exist, conns_exist))
    world_mapping = session.scalars(q).first()
    return world_mapping.from_dao() if world_mapping is not None else None


def main():
    semantic_world_database_uri = os.environ.get("SEMANTIC_WORLD_DATABASE_URI")

    # Create database engine and session
    engine = create_engine(f"mysql+pymysql://{semantic_world_database_uri}")
    session = Session(engine)

    # update schema
    drop_database(engine)
    Base.metadata.create_all(engine)

    parser = ProcTHORParser(
        "../../../../resources/procthor_json/house_987654321.json", session
    )
    world = parser.parse()

    rclpy.init()

    node = rclpy.create_node("viz_marker")

    p = VizMarkerPublisher(world, node)

    time.sleep(1000)
    p._stop_publishing()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
