import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable, Type

import numpy as np
import trimesh
from tqdm import tqdm

from ..spatial_types import Point3
from ..spatial_types.derivatives import DerivativeMap
from ..spatial_types.spatial_types import TransformationMatrix, Vector3
from ..views.factories import ViewFactory
from ..world import World
from ..world_description.connections import ActiveConnection, FixedConnection
from ..world_description.degree_of_freedom import DegreeOfFreedom
from ..world_description.geometry import TriangleMesh, FileMesh, Mesh
from ..world_description.world_entity import Body, Connection


@dataclass
class Step(ABC):
    """
    A Step is a transformation that takes a World as input and produces a modified World as output.
    Steps are intended to be used in a Pipeline, where the output World of one Step is passed as the input World to the next Step.
    """

    @abstractmethod
    def _apply(self, world: World) -> World: ...

    def apply(self, world: World) -> World:
        with world.modify_world():
            return self._apply(world)


@dataclass
class Pipeline:
    """
    A Pipeline is a sequence of Steps that are applied to a World in order.
    Each Step takes the World as input and produces a modified World as output.
    The output World of one Step is passed as the input World to the next Step.
    """

    steps: List[Step]
    """
    The list of Steps to be applied in the Pipeline.
    """

    def apply(self, world: World) -> World:
        for step in self.steps:
            world = step.apply(world)
        return world


@dataclass
class BodyFilter(Step):
    """
    Filters bodies in the world based on a given condition.
    """

    condition: Callable[[Body], bool]

    def _apply(self, world: World) -> World:
        for body in world.bodies:
            if not self.condition(body):
                world.remove_kinematic_structure_entity(body)
        return world


@dataclass
class CenterLocalGeometryAndPreserveWorldPose(Step):
    """
    Adjusts the vertices of the collision meshes of each body in the world so that the origin is at the center of the
    mesh, and then updates the parent connection of the body to preserve the original world pose.
    An example where this is useful is when parsing FBX files where all bodies in the resulting world have an origin
    at (0, 0, 0), even through the collision meshes are not centered around that point.
    """

    def _apply(self, world: World) -> World:
        for body in world.bodies_topologically_sorted:

            vertices = []

            for coll in body.collision:
                if isinstance(coll, (FileMesh, TriangleMesh)):
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
                if isinstance(coll, (FileMesh, TriangleMesh)):
                    m = coll.mesh
                    if m.vertices.shape[0] > 0:
                        m.vertices -= center

            old_origin_T_new_origin = TransformationMatrix.from_point_rotation_matrix(
                Point3(*center)
            )

            if body.parent_connection:
                parent_T_old_origin = body.parent_connection.origin_expression

                body.parent_connection.origin_expression = (
                    parent_T_old_origin @ old_origin_T_new_origin
                )

            for child in world.compute_child_kinematic_structure_entities(body):
                old_origin_T_child_origin = child.parent_connection.origin_expression
                child.parent_connection.origin_expression = (
                    old_origin_T_new_origin.inverse() @ old_origin_T_child_origin
                )
        return world


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

    def _apply(self, world: World) -> World:
        filtered_bodies = [body for body in world.bodies if self.body_condition(body)]

        for body in filtered_bodies:
            factory = self.factory_creator(body)
            parent_connection = body.parent_connection
            if parent_connection is None:
                return factory.create()

            for entity in world.compute_descendent_child_kinematic_structure_entities(
                body
            ):
                world.remove_kinematic_structure_entity(entity)

            world.remove_kinematic_structure_entity(body)

            new_world = factory.create()
            parent_connection.child = new_world.root
            world.merge_world(new_world, parent_connection)

        return world


@dataclass
class FillCollisionWithVisual(Step):
    """
    Fills the collision meshes of bodies in the world with their visual meshes if they have no collision meshes.
    """

    def _apply(self, world: World) -> World:
        for body in world.bodies:
            if len(body.collision) == 0 and len(body.visual) > 0:
                for vis in body.visual:
                    if isinstance(vis, Mesh):
                        body.collision.append(
                            TriangleMesh(
                                mesh=vis.mesh, scale=vis.scale, origin=vis.origin
                            )
                        )
        return world


@dataclass
class FilterOutBodiesByExactArea(Step):
    """
    Filters out bodies from the world whose total mesh area (collision or visual) matches a specified area.
    """

    area: float
    """
    The exact area to filter out bodies by.
    """

    def _apply(self, world: World) -> World:
        def body_area(body: Body) -> float:
            total_area = 0.0
            if len(body.collision) > 0:
                for coll in body.collision:
                    if isinstance(coll, Mesh):
                        mesh: trimesh.Trimesh = coll.mesh
                        if mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0:
                            total_area += mesh.area
            elif len(body.visual) > 0:
                for vis in body.visual:
                    if isinstance(vis, Mesh):
                        mesh: trimesh.Trimesh = vis.mesh
                        if mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0:
                            total_area += mesh.area
            return total_area

        bodies = list(world.bodies)

        for body in tqdm(bodies, desc="Filtering bodies by exact area"):
            if np.isclose(body_area(body), self.area):
                world.remove_kinematic_structure_entity(body)
        return world


@dataclass
class FilterOutLeafBodiesByMinArea(Step):
    """
    Filters out bodies from the world whose total mesh area (collision or visual) is below a specified minimum area.
    """

    min_area: float
    """
    The minimum area threshold to filter out bodies by.
    """

    def _apply(self, world: World) -> World:
        def body_area(body: Body) -> float:
            total_area = 0.0
            if len(body.collision) > 0:
                for coll in body.collision:
                    if isinstance(coll, Mesh):
                        mesh: trimesh.Trimesh = coll.mesh
                        total_area += mesh.area
            elif len(body.visual) > 0:
                for vis in body.visual:
                    if isinstance(vis, Mesh):
                        mesh: trimesh.Trimesh = vis.mesh
                        total_area += mesh.area
            return total_area

        bodies = list(world.bodies)

        for body in tqdm(bodies, desc="Filtering bodies by min area"):
            if len(world.compute_child_kinematic_structure_entities(body)) > 0:
                continue
            if body_area(body) < self.min_area:
                world.remove_kinematic_structure_entity(body)
        return world


@dataclass
class RemoveVisualsIfCondition(Step):
    """
    Removes visual meshes from bodies in the world that match a given condition.
    """

    condition: Callable[[Body], bool]

    def _apply(self, world: World) -> World:
        for body in world.bodies:
            if self.condition(body):
                body.visual.clear()
        return world


@dataclass
class ReplaceFixedConnectionWithActiveConnectionIfCondition(Step):
    """
    Replaces the parent connection of bodies in the world that match a given condition with a new active connection.
    """

    condition: Callable[[Body], bool]
    connection_type: Type[ActiveConnection]
    """
    A callable that takes no arguments and returns a new connection instance.
    """

    def _apply(self, world: World) -> World:
        lower_limits = DerivativeMap[float]()
        upper_limits = DerivativeMap[float]()

        lower_limits.position = 0
        upper_limits.position = np.pi / 1.5

        for body in world.bodies:
            if self.condition(body):
                parent_connection = body.parent_connection
                if parent_connection is None or not isinstance(
                    parent_connection, FixedConnection
                ):
                    continue
                dof = DegreeOfFreedom(
                    name=parent_connection.name,
                    lower_limits=lower_limits,
                    upper_limits=upper_limits,
                )
                parent = parent_connection.parent
                origin_expression = body.parent_connection.origin_expression
                new_connection = self.connection_type(
                    parent=parent,
                    child=body,
                    origin_expression=origin_expression,
                    multiplier=1.0,
                    offset=0.0,
                    axis=Vector3.Z(),
                    dof=dof,
                )
                world.remove_connection(parent_connection)
                world.add_connection(new_connection)
        return world
