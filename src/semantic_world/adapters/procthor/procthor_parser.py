import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
from random_events.polytope import Polytope
from random_events.variable import Continuous

from semantic_world.geometry import BoundingBoxCollection, Scale
from semantic_world.prefixed_name import PrefixedName
from semantic_world.spatial_types.spatial_types import TransformationMatrix, Point3
from semantic_world.variables import SpatialVariables
from semantic_world.views.factories import (
    DoorFactory,
    RoomFactory,
    WallFactory,
    HandleFactory,
    Direction,
)
from semantic_world.world_entity import Body, Region


@dataclass
class Wall:
    doors: List[dict] = field(default_factory=dict)
    walls: List[dict] = field(default_factory=list)


@dataclass
class ProcTHORParser:
    file_path: str

    def import_room(self, room) -> Tuple[RoomFactory, TransformationMatrix]:
        room_name = room["roomType"]
        room_id = room["id"].split("|")[-1]
        room_name = f"{room_name}_{room_id}"
        room_name = PrefixedName(room_name)

        room_polytope = room["floorPolygon"]

        room_polytope: List[Point3] = [
            Point3(v["x"], v["y"], v["z"]) for v in room_polytope
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

        room_factory = RoomFactory(name=room_name, polytope_corners=centered_polytope)

        transform = TransformationMatrix.from_xyz_rpy(
            center[0], center[1], center[2], 0, 0, 0
        )

        return room_factory, transform

    def import_object(self, parent_body, obj):
        body_name = obj["id"].replace("|", "_")

        asset_id = obj["assetId"]

        orm_query_object_using_asset_id = ...

        transform = TransformationMatrix.from_xyz_rpy(
            obj["position"]["x"],
            obj["position"]["y"],
            obj["position"]["z"],
            obj["rotation"]["x"],
            obj["rotation"]["y"],
            obj["rotation"]["z"],
        )

        for child in obj.get("children", {}):
            self.import_object(body_name, child)

        return

    @staticmethod
    def group_walls_by_polygon(
        remaining_walls: List[Dict],
    ) -> Tuple[List[Wall], List[Dict]]:
        """
        Groups walls with identical polygons (order-invariant) into pairs -> Wall([w1, w2]).
        Returns (paired_walls, leftovers) where leftovers are any unpaired walls.
        """

        def polygon_key(poly):
            # Treat as a set of points so ordering doesn’t matter
            return frozenset((p["x"], p["y"], p.get("z", 0)) for p in poly)

        groups: Dict[frozenset, List[Dict]] = {}
        for w in remaining_walls:
            key = polygon_key(w.get("polygon", []))
            groups.setdefault(key, []).append(w)

        paired: List["Wall"] = []
        leftovers: List[Dict] = []

        for walls in groups.values():
            i = 0
            while i + 1 < len(walls):
                paired.append(Wall(walls=[walls[i], walls[i + 1]]))
                i += 2
            if i < len(walls):
                leftovers.append(walls[i])

        return paired, leftovers

    def group_doors_with_walls(
        self, doors: List[Dict], walls: List[Dict]
    ) -> List[Wall]:
        """
        Returns:
          - door_groups: list of {'door': <door>, 'walls': [<wall0?>, <wall1?>], 'wallIds': [id0?, id1?]}
          - remaining_walls: walls not referenced by any door
        """

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
            door_groups.append(Wall(door=d, walls=found))

        remaining_walls = [w for w in walls if w["id"] not in used_wall_ids]

        paired_walls, unpaired_walls = self.group_walls_by_polygon(remaining_walls)

        door_groups.extend(paired_walls)

        assert (
            len(unpaired_walls) == 0
        ), "Apparently there are cases were there really is only one wall, not two with the same corners. This case may need to be handled now"

        return door_groups, paired_walls

    @staticmethod
    def polygon_scale_and_center(polygon):
        xs = [p["x"] for p in polygon]
        ys = [p["y"] for p in polygon]
        zs = [p["z"] for p in polygon]

        scale_x = max(xs) - min(xs)
        scale_y = max(ys) - min(ys)
        scale_z = max(zs) - min(zs)

        scale = Scale(scale_x, scale_y, scale_z)

        center_x = (max(xs) + min(xs)) / 2
        center_y = (max(ys) + min(ys)) / 2
        center_z = (max(zs) + min(zs)) / 2

        position = Point3(center_x, center_y, center_z)

        return position, scale

    def import_walls_with_doors(self, wall: Wall) -> Tuple[WallFactory, TransformationMatrix]:

        door_factories = []
        door_transforms = []

        for door in wall.walls:

            room_numbers = door["id"].split("|")[1:]

            door_name = PrefixedName(f"{door["assetId"]}_room{room_numbers[0]}_room{room_numbers[1]}")
            handle_name = PrefixedName(f"{door["assetId"]}_room{room_numbers[0]}_room{room_numbers[1]}_handle")

            door_position, door_scale = self.polygon_scale_and_center(door["holePolygon"])

            # I think a double door factory makes sense here, since it allows us to make assumptions about joints, scales, positions etc here
            handle_direction = Direction.Y
            door_factory = DoorFactory(
                name=door_name,
                scale=door_scale,
                handle_factory=HandleFactory(name=handle_name),
                handle_direction=handle_direction,
            )

            door_transform = TransformationMatrix.from_xyz_rpy(
                door_position.x, door_position.y, door_position.z, 0, 0, 0
            )

            door_factories.append(door_factory)
            door_transforms.append(door_transform)

        room_numbers = [w["id"].split("|")[1] for w in wall.walls]
        wall_name = PrefixedName(f"wall_room{room_numbers[0]}_room{room_numbers[1]}")

        wall_position, wall_scale = self.polygon_scale_and_center(
            wall.walls[0]["polygon"]
        )
        wall_factory = WallFactory(
            name=wall_name,
            scale=wall_scale,
            door_factories=door_factories,
            door_transforms=door_transforms,
        )

        wall_transform = TransformationMatrix.from_xyz_rpy(
            wall_position.x, wall_position.y, wall_position.z, 0, 0, 0
        )

        return wall_factory, wall_transform

    def parse(self):
        with open(self.file_path) as f:
            house = json.load(f)
        house_name = self.file_path.split("/")[-1].split(".")[0]

        rooms = house["rooms"]
        room_regions, room_transforms = [], []
        for room in rooms:
            self.import_room(room)

        objects = house["objects"]
        for obj in objects:
            self.import_object(house_name, obj)

        doors = house["doors"]
        walls = house["walls"]

        door_groups, remaining_walls = self.group_doors_with_walls(doors, walls)

        for door_group in door_groups:
            self.import_walls_with_doors(door_group)

        for wall in remaining_walls:
            self.import_wall(wall)


def main():
    parser = ProcTHORParser("../../../../resources/procthor_json/house_987654321.json")
    parser.parse()


if __name__ == "__main__":
    main()
