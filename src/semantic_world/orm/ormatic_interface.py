from semantic_world.orm.model import TransformationMatrixType

from sqlalchemy import Column, Float, ForeignKey, Integer, MetaData, String, Table
from sqlalchemy.orm import RelationshipProperty, registry, relationship
import semantic_world.connections
import semantic_world.geometry
import semantic_world.orm.model
import semantic_world.prefixed_name
import semantic_world.views.views
import semantic_world.world_entity

metadata = MetaData()


t_Color = Table(
    'Color', metadata,
    Column('id', Integer, primary_key=True),
    Column('R', Float, nullable=False),
    Column('G', Float, nullable=False),
    Column('B', Float, nullable=False),
    Column('A', Float, nullable=False)
)

t_PrefixedName = Table(
    'PrefixedName', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String(255), nullable=False),
    Column('prefix', String(255))
)

t_Scale = Table(
    'Scale', metadata,
    Column('id', Integer, primary_key=True),
    Column('x', Float, nullable=False),
    Column('y', Float, nullable=False),
    Column('z', Float, nullable=False)
)

t_UnitVector = Table(
    'UnitVector', metadata,
    Column('id', Integer, primary_key=True),
    Column('x', Float, nullable=False),
    Column('y', Float, nullable=False),
    Column('z', Float, nullable=False)
)

t_WorldEntity = Table(
    'WorldEntity', metadata,
    Column('id', Integer, primary_key=True),
    Column('polymorphic_type', String(255))
)

t_Body = Table(
    'Body', metadata,
    Column('id', ForeignKey('WorldEntity.id'), primary_key=True),
    Column('name_id', ForeignKey('PrefixedName.id'), nullable=False),
    Column('index', Integer)
)

t_View = Table(
    'View', metadata,
    Column('id', ForeignKey('WorldEntity.id'), primary_key=True)
)

t_ActiveConnection = Table(
    'ActiveConnection', metadata,
    Column('id', Integer, primary_key=True),
    Column('parent_id', ForeignKey('Body.id'), nullable=False),
    Column('child_id', ForeignKey('Body.id'), nullable=False),
    Column('polymorphic_type', String(255))
)

t_Components = Table(
    'Components', metadata,
    Column('id', ForeignKey('View.id'), primary_key=True)
)

t_Connection = Table(
    'Connection', metadata,
    Column('id', Integer, primary_key=True),
    Column('parent_id', ForeignKey('Body.id'), nullable=False),
    Column('child_id', ForeignKey('Body.id'), nullable=False)
)

t_Container = Table(
    'Container', metadata,
    Column('id', ForeignKey('View.id'), primary_key=True),
    Column('body_id', ForeignKey('Body.id'), nullable=False)
)

t_FixedConnection = Table(
    'FixedConnection', metadata,
    Column('id', Integer, primary_key=True),
    Column('parent_id', ForeignKey('Body.id'), nullable=False),
    Column('child_id', ForeignKey('Body.id'), nullable=False)
)

t_Fridge = Table(
    'Fridge', metadata,
    Column('id', ForeignKey('View.id'), primary_key=True),
    Column('body_id', ForeignKey('Body.id'), nullable=False)
)

t_Furniture = Table(
    'Furniture', metadata,
    Column('id', ForeignKey('View.id'), primary_key=True)
)

t_Handle = Table(
    'Handle', metadata,
    Column('id', ForeignKey('View.id'), primary_key=True),
    Column('body_id', ForeignKey('Body.id'), nullable=False)
)

t_PassiveConnection = Table(
    'PassiveConnection', metadata,
    Column('id', Integer, primary_key=True),
    Column('parent_id', ForeignKey('Body.id'), nullable=False),
    Column('child_id', ForeignKey('Body.id'), nullable=False),
    Column('polymorphic_type', String(255))
)

t_Shape = Table(
    'Shape', metadata,
    Column('id', Integer, primary_key=True),
    Column('origin', TransformationMatrixType, nullable=False),
    Column('body_visual_id', ForeignKey('Body.id')),
    Column('body_collision_id', ForeignKey('Body.id')),
    Column('polymorphic_type', String(255))
)

t_Connection6DoF = Table(
    'Connection6DoF', metadata,
    Column('id', ForeignKey('PassiveConnection.id'), primary_key=True)
)

t_Cupboard = Table(
    'Cupboard', metadata,
    Column('id', ForeignKey('Furniture.id'), primary_key=True)
)

t_Mesh = Table(
    'Mesh', metadata,
    Column('id', ForeignKey('Shape.id'), primary_key=True),
    Column('filename', String(255), nullable=False),
    Column('scale_id', ForeignKey('Scale.id'), nullable=False)
)

t_OmniDrive = Table(
    'OmniDrive', metadata,
    Column('id', ForeignKey('ActiveConnection.id'), primary_key=True),
    Column('translation_velocity_limits', Float, nullable=False),
    Column('rotation_velocity_limits', Float, nullable=False)
)

t_Primitive = Table(
    'Primitive', metadata,
    Column('id', ForeignKey('Shape.id'), primary_key=True),
    Column('color_id', ForeignKey('Color.id'), nullable=False)
)

t_PrismaticConnection = Table(
    'PrismaticConnection', metadata,
    Column('id', ForeignKey('ActiveConnection.id'), primary_key=True),
    Column('axis_id', ForeignKey('UnitVector.id'), nullable=False),
    Column('multiplier', Float, nullable=False),
    Column('offset', Float, nullable=False)
)

t_RevoluteConnection = Table(
    'RevoluteConnection', metadata,
    Column('id', ForeignKey('ActiveConnection.id'), primary_key=True),
    Column('axis_id', ForeignKey('UnitVector.id'), nullable=False),
    Column('multiplier', Float, nullable=False),
    Column('offset', Float, nullable=False)
)

t_Box = Table(
    'Box', metadata,
    Column('id', ForeignKey('Primitive.id'), primary_key=True),
    Column('scale_id', ForeignKey('Scale.id'), nullable=False)
)

t_Cabinet = Table(
    'Cabinet', metadata,
    Column('id', ForeignKey('Cupboard.id'), primary_key=True),
    Column('container_id', ForeignKey('Container.id'), nullable=False)
)

t_Cylinder = Table(
    'Cylinder', metadata,
    Column('id', ForeignKey('Primitive.id'), primary_key=True),
    Column('width', Float, nullable=False),
    Column('height', Float, nullable=False)
)

t_Sphere = Table(
    'Sphere', metadata,
    Column('id', ForeignKey('Primitive.id'), primary_key=True),
    Column('radius', Float, nullable=False)
)

t_Wardrobe = Table(
    'Wardrobe', metadata,
    Column('id', ForeignKey('Cupboard.id'), primary_key=True)
)

t_Door = Table(
    'Door', metadata,
    Column('id', ForeignKey('Components.id'), primary_key=True),
    Column('body_id', ForeignKey('Body.id'), nullable=False),
    Column('handle_id', ForeignKey('Handle.id'), nullable=False),
    Column('wardrobe_doors_id', ForeignKey('Wardrobe.id'))
)

t_Drawer = Table(
    'Drawer', metadata,
    Column('id', ForeignKey('Components.id'), primary_key=True),
    Column('container_id', ForeignKey('Container.id'), nullable=False),
    Column('handle_id', ForeignKey('Handle.id'), nullable=False),
    Column('cabinet_drawers_id', ForeignKey('Cabinet.id'))
)

mapper_registry = registry(metadata=metadata)

m_Color = mapper_registry.map_imperatively(semantic_world.geometry.Color, t_Color, )

m_Scale = mapper_registry.map_imperatively(semantic_world.geometry.Scale, t_Scale, )

m_Connection = mapper_registry.map_imperatively(semantic_world.world_entity.Connection, t_Connection, properties = dict(parent=relationship('Body',foreign_keys=[t_Connection.c.parent_id]), 
child=relationship('Body',foreign_keys=[t_Connection.c.child_id])))

m_Shape = mapper_registry.map_imperatively(semantic_world.geometry.Shape, t_Shape, properties = dict(origin=t_Shape.c.origin), polymorphic_on = "polymorphic_type", polymorphic_identity = "Shape")

m_WorldEntity = mapper_registry.map_imperatively(semantic_world.world_entity.WorldEntity, t_WorldEntity, polymorphic_on = "polymorphic_type", polymorphic_identity = "WorldEntity")

m_FixedConnection = mapper_registry.map_imperatively(semantic_world.connections.FixedConnection, t_FixedConnection, properties = dict(parent=relationship('Body',foreign_keys=[t_FixedConnection.c.parent_id]), 
child=relationship('Body',foreign_keys=[t_FixedConnection.c.child_id])))

m_ActiveConnection = mapper_registry.map_imperatively(semantic_world.connections.ActiveConnection, t_ActiveConnection, properties = dict(parent=relationship('Body',foreign_keys=[t_ActiveConnection.c.parent_id]), 
child=relationship('Body',foreign_keys=[t_ActiveConnection.c.child_id])), polymorphic_on = "polymorphic_type", polymorphic_identity = "ActiveConnection")

m_PassiveConnection = mapper_registry.map_imperatively(semantic_world.connections.PassiveConnection, t_PassiveConnection, properties = dict(parent=relationship('Body',foreign_keys=[t_PassiveConnection.c.parent_id]), 
child=relationship('Body',foreign_keys=[t_PassiveConnection.c.child_id])), polymorphic_on = "polymorphic_type", polymorphic_identity = "PassiveConnection")

m_UnitVector = mapper_registry.map_imperatively(semantic_world.connections.UnitVector, t_UnitVector, )

m_PrefixedName = mapper_registry.map_imperatively(semantic_world.prefixed_name.PrefixedName, t_PrefixedName, )

m_Primitive = mapper_registry.map_imperatively(semantic_world.geometry.Primitive, t_Primitive, properties = dict(color=relationship('Color',foreign_keys=[t_Primitive.c.color_id])), polymorphic_identity = "Primitive", inherits = m_Shape)

m_Mesh = mapper_registry.map_imperatively(semantic_world.geometry.Mesh, t_Mesh, properties = dict(scale=relationship('Scale',foreign_keys=[t_Mesh.c.scale_id])), polymorphic_identity = "Mesh", inherits = m_Shape)

m_View = mapper_registry.map_imperatively(semantic_world.world_entity.View, t_View, polymorphic_identity = "View", inherits = m_WorldEntity)

m_Body = mapper_registry.map_imperatively(semantic_world.world_entity.Body, t_Body, properties = dict(name=relationship('PrefixedName',foreign_keys=[t_Body.c.name_id]), 
visual=relationship('Shape',foreign_keys=[t_Shape.c.body_visual_id]), 
collision=relationship('Shape',foreign_keys=[t_Shape.c.body_collision_id])), polymorphic_identity = "Body", inherits = m_WorldEntity)

m_PrismaticConnection = mapper_registry.map_imperatively(semantic_world.connections.PrismaticConnection, t_PrismaticConnection, properties = dict(axis=relationship('UnitVector',foreign_keys=[t_PrismaticConnection.c.axis_id])), polymorphic_identity = "PrismaticConnection", inherits = m_ActiveConnection)

m_RevoluteConnection = mapper_registry.map_imperatively(semantic_world.connections.RevoluteConnection, t_RevoluteConnection, properties = dict(axis=relationship('UnitVector',foreign_keys=[t_RevoluteConnection.c.axis_id])), polymorphic_identity = "RevoluteConnection", inherits = m_ActiveConnection)

m_OmniDrive = mapper_registry.map_imperatively(semantic_world.connections.OmniDrive, t_OmniDrive, polymorphic_identity = "OmniDrive", inherits = m_ActiveConnection)

m_Connection6DoF = mapper_registry.map_imperatively(semantic_world.connections.Connection6DoF, t_Connection6DoF, polymorphic_identity = "Connection6DoF", inherits = m_PassiveConnection)

m_Sphere = mapper_registry.map_imperatively(semantic_world.geometry.Sphere, t_Sphere, polymorphic_identity = "Sphere", inherits = m_Primitive)

m_Cylinder = mapper_registry.map_imperatively(semantic_world.geometry.Cylinder, t_Cylinder, polymorphic_identity = "Cylinder", inherits = m_Primitive)

m_Box = mapper_registry.map_imperatively(semantic_world.geometry.Box, t_Box, properties = dict(scale=relationship('Scale',foreign_keys=[t_Box.c.scale_id])), polymorphic_identity = "Box", inherits = m_Primitive)

m_Furniture = mapper_registry.map_imperatively(semantic_world.views.views.Furniture, t_Furniture, polymorphic_identity = "Furniture", inherits = m_View)

m_Fridge = mapper_registry.map_imperatively(semantic_world.views.views.Fridge, t_Fridge, properties = dict(body=relationship('Body',foreign_keys=[t_Fridge.c.body_id])), polymorphic_identity = "Fridge", inherits = m_View)

m_Components = mapper_registry.map_imperatively(semantic_world.views.views.Components, t_Components, polymorphic_identity = "Components", inherits = m_View)

m_Handle = mapper_registry.map_imperatively(semantic_world.views.views.Handle, t_Handle, properties = dict(body=relationship('Body',foreign_keys=[t_Handle.c.body_id])), polymorphic_identity = "Handle", inherits = m_View)

m_Container = mapper_registry.map_imperatively(semantic_world.views.views.Container, t_Container, properties = dict(body=relationship('Body',foreign_keys=[t_Container.c.body_id])), polymorphic_identity = "Container", inherits = m_View)

m_Cupboard = mapper_registry.map_imperatively(semantic_world.views.views.Cupboard, t_Cupboard, polymorphic_identity = "Cupboard", inherits = m_Furniture)

m_Drawer = mapper_registry.map_imperatively(semantic_world.views.views.Drawer, t_Drawer, properties = dict(container=relationship('Container',foreign_keys=[t_Drawer.c.container_id]), 
handle=relationship('Handle',foreign_keys=[t_Drawer.c.handle_id])), polymorphic_identity = "Drawer", inherits = m_Components)

m_Door = mapper_registry.map_imperatively(semantic_world.views.views.Door, t_Door, properties = dict(body=relationship('Body',foreign_keys=[t_Door.c.body_id]), 
handle=relationship('Handle',foreign_keys=[t_Door.c.handle_id])), polymorphic_identity = "Door", inherits = m_Components)

m_Cabinet = mapper_registry.map_imperatively(semantic_world.views.views.Cabinet, t_Cabinet, properties = dict(container=relationship('Container',foreign_keys=[t_Cabinet.c.container_id]), 
drawers=relationship('Drawer',foreign_keys=[t_Drawer.c.cabinet_drawers_id])), polymorphic_identity = "Cabinet", inherits = m_Cupboard)

m_Wardrobe = mapper_registry.map_imperatively(semantic_world.views.views.Wardrobe, t_Wardrobe, properties = dict(doors=relationship('Door',foreign_keys=[t_Door.c.wardrobe_doors_id])), polymorphic_identity = "Wardrobe", inherits = m_Cupboard)
