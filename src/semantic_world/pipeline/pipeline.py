import logging
import re
from abc import ABC
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional, Callable

import coacd
import numpy as np
import trimesh

from ..spatial_types import Point3
from ..spatial_types.spatial_types import TransformationMatrix
from ..views.factories import ViewFactory
from ..world import World
from ..world_description.geometry import TriangleMesh, Mesh
from ..world_description.world_entity import Body


@dataclass
class Step(ABC):
    """
    A Step is a transformation that takes a World as input and produces a modified World as output.
    Steps are intended to be used in a Pipeline, where the output World of one Step is passed as the input World to the next Step.
    Steps modify the World in-place, and return the modified World.
    """

    world: Optional[World] = field(init=False, default=None)
    """
    World to be transformed by this Step. This is set when the Step is applied in a Pipeline.
    """

    def _apply(self) -> World:
        raise NotImplementedError()

    def apply(self) -> World:
        with self.world.modify_world():
            return self._apply()


@dataclass
class Pipeline:
    """
    A Pipeline is a sequence of Steps that are applied to a World in order, in-place.
    Each Step takes the World as input and produces a modified World as output.
    The output World of one Step is passed as the input World to the next Step.
    """

    steps: List[Step]
    """
    The list of Steps to be applied in the Pipeline.
    """

    def apply(self, world: World) -> World:
        for step in self.steps:
            step.world = world
            world = step.apply()
        return world


@dataclass
class BodyFilter(Step):
    """
    Filters bodies in the world based on a given condition.
    """

    condition: Callable[[Body], bool]

    def _apply(self) -> World:
        for body in self.world.bodies:
            if not self.condition(body):
                self.world.remove_kinematic_structure_entity(body)
        return self.world


@dataclass
class CenterLocalGeometryPreserveWorldPose(Step):
    """
    Adjusts the vertices of the collision meshes of each body in the world so that the origin is at the center of the
    mesh, and then updates the parent connection of the body to preserve the original world pose.
    An example where this is useful is when parsing FBX files where all bodies in the resulting world have an origin
    at (0, 0, 0), even through the collision meshes are not centered around that point.
    """

    def _apply(self) -> World:
        for body in self.world.bodies_topologically_sorted:

            vertices = []

            for coll in body.collision:
                if isinstance(coll, (Mesh, TriangleMesh)):
                    mesh = coll.mesh
                    if mesh.vertices.shape[0] > 0:
                        vertices.append(mesh.vertices.copy())

            if len(vertices) == 0:
                logging.warning(
                    f"Body {body.name.name} has no vertices in visual or collision shapes, skipping."
                )
                continue

            # Compute the axis-aligned bounding box center of all vertices
            all_vertices = np.vstack(vertices)
            mins = all_vertices.min(axis=0)
            maxs = all_vertices.max(axis=0)
            center = (mins + maxs) / 2.0

            for coll in body.collision:
                if isinstance(coll, (Mesh, TriangleMesh)):
                    m = coll.mesh
                    if m.vertices.shape[0] > 0:
                        m.vertices -= center

            old_origin_T_new_origin = TransformationMatrix.from_point_rotation_matrix(
                Point3(*center)
            )

            parent_T_old_origin = body.parent_connection.origin_expression

            body.parent_connection.origin_expression = (
                parent_T_old_origin @ old_origin_T_new_origin
            )

            for child in self.world.compute_child_kinematic_structure_entities(body):
                old_origin_T_child_origin = child.parent_connection.origin_expression
                child.parent_connection.origin_expression = (
                    old_origin_T_new_origin.inverse() @ old_origin_T_child_origin
                )

        self.world._notify_model_change()
        return self.world


class ApproximationMode(StrEnum):
    """
    Approximation shape type
    """

    BOX = "box"
    CONVEX_HULL = "ch"


class PreprocessingMode(StrEnum):
    """
    Manifold preprocessing mode
    """

    AUTO = "auto"
    """
    Automatically chose based on the geometry.
    """

    ON = "on"
    """
    Force turn on the pre-processing
    """

    OFF = "off"
    """
    Force turn off the pre-processing
    """


@dataclass
class COACDMeshDecomposer(Step):
    """
    COACDMeshDecomposer is a class for decomposing complex 3D meshes into simpler convex components
    using the COACD (Convex Optimization for Approximate Convex Decomposition) algorithm. It is
    designed to preprocess, analyze, and process 3D meshes with a focus on efficiency and scalability
    in fields such as robotics, gaming, and simulation.

    Check https://github.com/SarahWeiii/CoACD for further details.
    """

    threshold: float = 0.05
    """
    Concavity threshold for terminating the decomposition (0.01 - 1)
    """

    max_convex_hull: Optional[int] = None
    """
    Maximum number of convex hulls in the result. 
    Works only when merge is enabled (may introduce convex hull with a concavity larger than the threshold)
    """

    preprocess_mode: PreprocessingMode = PreprocessingMode.AUTO
    """
    Manifold preprocessing mode.
    """

    preprocess_resolution: int = 50
    """
    Resolution for manifold preprocess (20~100)
    """

    resolution: int = 2000
    """
    Sampling resolution for Hausdorff distance calculation (1 000 - 10 000)
    """

    search_nodes: int = 20
    """
    Max number of child nodes in the monte carlo tree search (10 - 40).
    """

    search_iterations: int = 150
    """
    Number of search iterations in the monte carlo tree search (60 - 2000).
    """

    search_depth: int = 3
    """
    Maximum search depth in the monte carlo tree search (2 - 7).
    """

    pca: bool = False
    """
    Enable PCA pre-processing
    """

    merge: bool = True
    """
    Enable merge postprocessing.
    """

    max_convex_hull_vertices: Optional[int] = None
    """
    Maximum vertex value for each convex hull, only when decimate is enabled.
    """

    extrude_margin: Optional[float] = None
    """
    Extrude margin, only when extrude is enabled
    """

    approximation_mode: ApproximationMode = ApproximationMode.BOX
    """
    Approximation mode to use.
    """

    seed: int = field(default_factory=lambda: np.random.randint(2**32))
    """
    Random seed used for sampling.
    """

    def _apply(self) -> World:
        for body in self.world.bodies:
            new_geometry = []

            for shape in body.visual:
                if isinstance(shape, (Mesh, TriangleMesh)):
                    mesh = shape.mesh

                    if shape.scale.x == shape.scale.y == shape.scale.z:
                        mesh.apply_scale(shape.scale.x)
                    else:
                        logging.warning(
                            "Ambiguous scale for mesh, using uniform scale only."
                        )

                    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
                    if self.max_convex_hull is not None:
                        max_convex_hull = self.max_convex_hull
                    else:
                        max_convex_hull = -1
                    parts = coacd.run_coacd(
                        mesh=mesh,
                        apx_mode=str(self.approximation_mode),
                        threshold=self.threshold,
                        max_convex_hull=max_convex_hull,
                        preprocess_mode=str(self.preprocess_mode),
                        resolution=self.resolution,
                        mcts_nodes=self.search_nodes,
                        mcts_iterations=self.search_iterations,
                        mcts_max_depth=self.search_depth,
                        pca=self.pca,
                        merge=self.merge,
                        decimate=self.max_convex_hull_vertices is not None,
                        max_ch_vertex=self.max_convex_hull_vertices or 256,
                        extrude=self.extrude_margin is not None,
                        extrude_margin=self.extrude_margin or 0.01,
                        seed=self.seed,
                    )

                    for vs, fs in parts:
                        new_geometry.append(
                            TriangleMesh(
                                mesh=trimesh.Trimesh(vs, fs), origin=shape.origin
                            )
                        )
                else:
                    new_geometry.append(shape)

            body.collision = new_geometry

        return self.world


@dataclass
class BodyFactoryReplace(Step):
    """
    Replace bodies in the world that match a given condition with new structures created by a factory.
    """

    body_condition: Callable[[Body], bool] = lambda x: bool(
        re.compile(r"^dresser_\d+.*$").fullmatch(x.name.name)
    )
    """
    Condition to filter bodies that should be replaced. Defaults to matching bodies containing "dresser_" followed by digits in their name.
    """

    factory_creator: Callable[[Body], ViewFactory] = None
    """
    A callable that takes a Body and returns a ViewFactory to create the new structure.
    """

    def _apply(self) -> World:
        filtered_bodies = [
            body for body in self.world.bodies if self.body_condition(body)
        ]

        for body in filtered_bodies:
            factory = self.factory_creator(body)
            parent_connection = body.parent_connection
            if parent_connection is None:
                return factory.create()

            for (
                entity
            ) in self.world.compute_descendent_child_kinematic_structure_entities(body):
                self.world.remove_kinematic_structure_entity(entity)

            self.world.remove_kinematic_structure_entity(body)

            new_world = factory.create()
            parent_connection.child = new_world.root
            self.world.merge_world(new_world, parent_connection)

        return self.world
