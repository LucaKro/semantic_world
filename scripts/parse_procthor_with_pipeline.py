import os
import re
import time
from functools import reduce
from operator import or_

import plotly.graph_objects as go
import tqdm
from ormatic.dao import to_dao
from ormatic.utils import drop_database
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from semantic_world.adapters.fbx import FBXParser as BlenderFBXParser
from semantic_world.adapters.mesh import FBXParser
from semantic_world.adapters.procthor.procthor_pipelines import dresser_factory_replace
from semantic_world.orm.ormatic_interface import *
from semantic_world.pipeline.pipeline import Pipeline, BodyFilter, BodyFactoryReplace, \
    CenterLocalGeometryPreserveWorldPose


def parse_fbx_file(fbx_file, session):
    dresser_pattern = re.compile(r"^.*dresser_(?!drawer\b).*$", re.IGNORECASE)

    pipeline = Pipeline(
        [
            BodyFilter(lambda x: not x.name.name.startswith("PS_")),
            # COACDMeshDecomposer(search_iterations=60, max_convex_hull=1)
            BodyFactoryReplace(
                body_condition=lambda b: bool(dresser_pattern.fullmatch(b.name.name)),
                factory_creator=dresser_factory_replace,
            ),
        ]
    )
    parser = BlenderFBXParser(fbx_file)
    worlds = parser.parse()

    worlds = [
        pipeline.apply(world)
        for world in worlds
        # if world.root.name.name == "dresser_413"
    ]
    if worlds:
        # events = [
        #     b.as_bounding_box_collection(worlds[0].root).event for b in worlds[0].bodies
        # ]
        # event = reduce(or_, events)

        # go.Figure(event.plot()).show()
        daos = [to_dao(world) for world in worlds]

        session.add_all(daos)

def parse_fbx_file_new(fbx_file, session):
    dresser_pattern = re.compile(r"^.*dresser_(?!drawer\b).*$", re.IGNORECASE)

    pipeline = Pipeline(
        [
            CenterLocalGeometryPreserveWorldPose(),
            BodyFilter(lambda x: not x.name.name.startswith("PS_")),
            # COACDMeshDecomposer(search_iterations=60, max_convex_hull=1)
            BodyFactoryReplace(
                body_condition=lambda b: bool(dresser_pattern.fullmatch(b.name.name)) and not ("drawer" in b.name.name.lower() or "door" in b.name.name.lower()),
                factory_creator=dresser_factory_replace,
            ),
        ]
    )

    parser = FBXParser(fbx_file)
    world = parser.parse()

    world = pipeline.apply(world)
    root_children = world.compute_child_bodies(world.root)

    with world.modify_world():

        worlds = [
            world.move_subgraph_from_root_to_new_world(child) for child in root_children
        ]

    if worlds:
        # events = [
        #     b.as_bounding_box_collection(worlds[0].root).event for b in worlds[0].bodies
        # ]
        # event = reduce(or_, events)

        # go.Figure(event.plot()).show()
        daos = [to_dao(world) for world in worlds]

        session.add_all(daos)


def main():
    semantic_world_database_uri = os.environ.get("SEMANTIC_WORLD_DATABASE_URI")
    procthor_root = os.path.join(os.path.expanduser("~"), "ai2thor")
    procthor_root = os.path.join(os.path.expanduser("~"), "work", "ai2thor")

    files = []
    for root, dirs, filenames in os.walk(procthor_root):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    pattern = re.compile(r".*_grp\.fbx$", re.IGNORECASE)

    excluded_words = [
        "FirstPersonCharacter",
        "SourceFiles_Procedural",
        "RobotArmTest",
        "_shards_",
    ]

    fbx_files = [
        f
        for f in files
        if not any([e in f for e in excluded_words]) and pattern.fullmatch(f)
    ]
    # Create database engine and session
    engine = create_engine(f"mysql+pymysql://{semantic_world_database_uri}")
    session = Session(engine)

    # update schema
    drop_database(engine)
    Base.metadata.create_all(engine)

    start_time = time.time_ns()
    for fbx_file in tqdm.tqdm(fbx_files):
        # if not 'dressers_grp' in fbx_file:
        #     continue
        parse_fbx_file_new(fbx_file, session)
    session.commit()
    print(
        f"Parsing {len(fbx_files)} files took {time.time_ns() - start_time} ns. In seconds: {(time.time_ns() - start_time) / 1e9}"
    )
    #
    # world = session.scalars(select(WorldMappingDAO)).one()
    # world = world.from_dao()


if __name__ == "__main__":
    main()
