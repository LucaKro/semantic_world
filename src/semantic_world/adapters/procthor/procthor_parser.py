import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# Python
from typing import Optional

import numpy as np
import rclpy
from ormatic.utils import drop_database
from random_events.interval import closed
from random_events.polytope import Polytope
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
from sqlalchemy import create_engine
from sqlalchemy import select, exists, and_, or_
from sqlalchemy.orm import Session
from scipy.spatial.transform import Rotation

from semantic_world.adapters.viz_marker import VizMarkerPublisher
from semantic_world.connections import FixedConnection
from semantic_world.geometry import Scale, BoundingBoxCollection
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
    Quaternion,
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
class Wall:
    doors: List[dict] = field(default_factory=dict)
    walls: List[dict] = field(default_factory=list)


def unity4x4_to_target4x4(M_u):
    """
    Convert a 4x4 Unity transform to the right-handed Z-up, X-forward system.
    M_u: (4,4) float-like
    returns: (4,4) numpy.ndarray
    """
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

    M_u = np.asarray(M_u, float).reshape(4, 4)

    return conjugation_matrix @ M_u @ inverse_conjugation_matrix


@dataclass
class ProcTHORParser:
    file_path: str

    session: Session

    def import_room(self, room) -> Tuple[World, TransformationMatrix]:
        room_name = room["roomType"]
        room_id = room["id"].split("|")[-1]
        room_name = f"{room_name}_{room_id}"
        room_name = PrefixedName(room_name)
        reference_frame = Body(name=room_name)

        room_polytopes = room["floorPolygon"]

        room_polytope: List[Point3] = [
            Point3(v["z"], -v["x"], v["y"]) for v in room_polytopes
        ]

        polytope_length = len(room_polytope)
        cx = sum(v.x.to_np() for v in room_polytope) / polytope_length
        cy = sum(v.y.to_np() for v in room_polytope) / polytope_length
        cz = sum(v.z.to_np() for v in room_polytope) / polytope_length

        center = (cx, cy, cz)

        centered_polytope = [
            Point3(v.x.to_np() - cx, v.y.to_np() - cy, v.z.to_np() - cz)
            for v in room_polytope
        ]

        min_height = min(v.y.to_np() for v in room_polytope)
        max_height = max(v.y.to_np() for v in room_polytope)

        points_2d = np.array([[p.x.to_np(), p.y.to_np()] for p in centered_polytope])
        polytope = Polytope.from_2d_points(points_2d)
        region_event = polytope.maximum_inner_box().to_simple_event().as_composite_set()

        region_event = region_event.update_variables(
            {
                Continuous("x_0"): SpatialVariables.x.value,
                Continuous("x_1"): SpatialVariables.y.value,
            }
        )
        region_event.fill_missing_variables([SpatialVariables.z.value])
        floor_event = SimpleEvent(
            {
                SpatialVariables.z.value: closed(min_height - 0.01, max_height + 0.01),
            }
        ).as_composite_set()
        floor_event.fill_missing_variables(SpatialVariables.xz)

        region_event = region_event & floor_event

        region_bb_collection = BoundingBoxCollection.from_event(region_event)

        region_shapes = region_bb_collection.as_shapes(reference_frame=reference_frame)

        region = Region(
            name=PrefixedName(room_name.name + "_region"),
            areas=region_shapes,
            reference_frame=reference_frame,
        )

        room_factory = RoomFactory(name=room_name, region=region)

        transform = TransformationMatrix.from_xyz_rpy(
            center[0], center[1], center[2], 0, 0, 0
        )

        return room_factory.create(), transform

    def import_object(self, obj) -> Tuple[World, TransformationMatrix]:
        asset_id = obj["assetId"]

        body_world: World = get_world_by_prefixed_name(self.session, name=asset_id)

        if body_world is None:
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

        body_transform = unity4x4_to_target4x4(body_transform)

        rotation = Rotation.from_matrix(body_transform[:3, :3]).as_quat()

        body_transform = TransformationMatrix.from_point_rotation_matrix(
            Point3(*body_transform[:3, 3]),
            RotationMatrix.from_quaternion(Quaternion.from_iterable(rotation)),
        )

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
        remaining_walls: List[Dict],
    ) -> List[Wall]:
        """
        Groups walls with identical polygons (order-invariant) into pairs -> Wall([w1, w2]).
        Returns (paired_walls, leftovers) where leftovers are any unpaired walls.
        """

        def polygon_key(poly):
            # Treat as a set of points so ordering doesn’t matter
            return frozenset((p["x"], p["y"], p["z"]) for p in poly)

        groups: Dict[frozenset, List[Dict]] = {}
        for w in remaining_walls:
            key = polygon_key(w.get("polygon", []))
            groups.setdefault(key, []).append(w)

        paired: List[Wall] = []
        for walls in groups.values():
            i = 0
            while i + 1 < len(walls):
                paired.append(Wall(walls=[walls[i], walls[i + 1]]))
                i += 2
            if i < len(walls):
                raise AssertionError(
                    "Apparently there are cases were there really is only one wall, not two with the same corners. This case may need to be handled now"
                )

        return paired

    def group_doors_with_walls(
        self, doors: List[Dict], walls: List[Dict]
    ) -> List[Wall]:
        walls_by_id = {w["id"]: w for w in walls}
        used_wall_ids = set()
        door_groups = []

        for d in doors:
            ids = [d.get("wall0"), d.get("wall1")]
            found = []
            for wid in ids:
                if not wid:
                    continue
                w = walls_by_id.get(wid)
                if w:
                    found.append(w)
                    used_wall_ids.add(wid)
            door_groups.append(Wall(doors=[d], walls=found))

        remaining_walls = [w for w in walls if w["id"] not in used_wall_ids]

        paired_walls = self.group_walls_by_polygon(remaining_walls)

        door_groups.extend(paired_walls)

        return door_groups

    @staticmethod
    def get_polygon_scale_and_center(polygon):
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
        cx = (x1 + x2) * 0.5
        cy = (min_y + max_y) * 0.5
        cz = (z1 + z2) * 0.5

        # Yaw of outward normal (Unity LH): n = up × along = (dz, 0, -dx)
        yaw = math.atan2(dz, -dx)

        thickness = 0.02
        scale = Scale(x=width, y=height, z=thickness)
        transform = TransformationMatrix.from_xyz_rpy(cx, cy, cz, 0.0, yaw, 0)
        return scale, transform

    def import_walls(self, wall: Wall) -> Tuple[World, TransformationMatrix]:

        room_numbers = [w["id"].split("|")[1] for w in wall.walls]
        wall_corners = [w["id"].split("|")[2:] for w in wall.walls][0]
        wall_name = PrefixedName(
            f"wall_{wall_corners[0]}_{wall_corners[1]}_{wall_corners[2]}_{wall_corners[3]}_room{room_numbers[0]}_room{room_numbers[1]}"
        )

        idx = 0 if room_numbers[0] > room_numbers[1] else 1

        wall_scale, old_wall_transform = self.get_polygon_scale_and_center(
            wall.walls[idx]["polygon"]
        )

        wall_scale = Scale(wall_scale.z, wall_scale.x, wall_scale.y)

        wall_transform = unity4x4_to_target4x4(old_wall_transform.to_np())

        position = wall_transform[:3, 3]
        rotation = Rotation.from_matrix(wall_transform[:3, :3]).as_quat()

        wall_transform = TransformationMatrix.from_point_rotation_matrix(
            Point3(position[0], position[1], 0),
            RotationMatrix.from_quaternion(Quaternion.from_iterable(rotation)),
        )

        door_factories = []
        door_transforms = []

        for door in wall.doors:
            room_numbers = door["id"].split("|")[1:]

            door_name = PrefixedName(
                f"{door["assetId"]}_room{room_numbers[0]}_room{room_numbers[1]}"
            )
            handle_name = PrefixedName(
                f"{door["assetId"]}_room{room_numbers[0]}_room{room_numbers[1]}_handle"
            )

            door_scale, old_door_transform = self.get_polygon_scale_and_center(
                door["holePolygon"]
            )

            door_scale = Scale(door_scale.z, door_scale.x, door_scale.y)

            door_position = old_door_transform.to_position().to_np()

            door_position = Point3(door_position[2], -door_position[0], door_position[1])

            relative_door_position = door_position - wall_transform.to_position()
            # door_transform = unity4x4_to_target4x4(door_transform.to_np())

            # rotation = Rotation.from_matrix(door_transform[:3, :3]).as_quat()

            door_transform = TransformationMatrix.from_point_rotation_matrix(
                Point3(0, relative_door_position.y, relative_door_position.z),
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
        with open(self.file_path) as f:
            house = json.load(f)
        house_name = self.file_path.split("/")[-1].split(".")[0]

        world = World(name=house_name)
        world_root = Body(name=PrefixedName(house_name))
        world.add_body(world_root)

        rooms = house["rooms"]
        for room in rooms:
            room_world, room_transform = self.import_room(room)
            room_connection = FixedConnection(
                parent=world.root,
                child=room_world.root,
                origin_expression=room_transform,
            )
            world.merge_world(room_world, room_connection)

        objects = house["objects"]
        for obj in objects:
            obj_world, obj_transform = self.import_object(obj)
            obj_connection = FixedConnection(
                parent=world.root, child=obj_world.root, origin_expression=obj_transform
            )
            world.merge_world(obj_world, obj_connection)

        doors = house["doors"]
        walls = house["walls"]

        walls = self.group_doors_with_walls(doors, walls)

        for index, wall in enumerate(walls):
            if index == 2:
                wall_world, wall_transform = self.import_walls(wall)
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
