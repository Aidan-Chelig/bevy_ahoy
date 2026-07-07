//! Box3D interop and a lightweight Bevy 0.18 runtime.
//!
//! This module is enabled with the `box3d` feature and re-exports the
//! low-level [`box3d`] crate and the Bevy adapter [`bevy_box3d`].
//!
//! The current Ahoy controller implementation still runs on Avian because it
//! depends on Avian's move-and-slide queries and collider metadata. At the time
//! this feature was added, `bevy_box3d` targets Bevy 0.19 while Ahoy targets
//! Bevy 0.18. [`AhoyBox3dPlugin`] provides a small native Box3D runtime that
//! compiles on Ahoy's Bevy version; [`bevy_box3d`] is re-exported for projects
//! that can use its Bevy 0.19 integration directly.

use bevy_app::{FixedUpdate, Plugin, PostUpdate};
use bevy_ecs::{
    prelude::*,
    system::{NonSend, NonSendMut},
};
use bevy_math::{Quat, Vec3};
use bevy_transform::prelude::Transform;
use std::collections::HashMap;

pub use ::bevy_box3d;
pub use ::box3d;

/// Common Box3D imports for users experimenting with the `box3d` feature.
pub mod prelude {
    pub use super::{
        AhoyBox3dBody, AhoyBox3dCollider, AhoyBox3dConfig, AhoyBox3dPlugin, AhoyBox3dShape,
        AhoyBox3dVelocity, Box3dBodyType, Box3dColliderShape, Box3dRuntime,
        bevy_box3d::{
            Box3dBody, Box3dConfig, Box3dContactEnded, Box3dContactHit, Box3dContactStarted,
            Box3dDebugConfig, Box3dDebugPlugin, Box3dPlugin, Box3dSensorEnded, Box3dSensorStarted,
            Box3dSet, Box3dShape, Box3dStats, Box3dWorld, Collider, ColliderParent, Damping,
            RigidBody, SleepThreshold, Velocity,
        },
        box3d::{
            BodyDef, BodyId, BodyType, CollisionPlane, Filter, MoverCapsule, QueryFilter, ShapeDef,
            ShapeId, ShapeProxy, SurfaceMaterial, World, clip_vector, solve_planes,
        },
    };
}

/// Bevy 0.18-compatible Box3D runtime plugin.
#[derive(Clone, Copy, Debug)]
pub struct AhoyBox3dPlugin {
    pub config: AhoyBox3dConfig,
}

impl Default for AhoyBox3dPlugin {
    fn default() -> Self {
        Self {
            config: AhoyBox3dConfig::default(),
        }
    }
}

impl Plugin for AhoyBox3dPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(self.config)
            .insert_non_send_resource(Box3dRuntime::new(self.config))
            .add_systems(
                FixedUpdate,
                (
                    cleanup_box3d_shapes,
                    cleanup_box3d_bodies,
                    create_box3d_bodies,
                    create_box3d_shapes,
                    sync_box3d_body_changes,
                    sync_box3d_velocity_changes,
                    step_box3d,
                )
                    .chain(),
            )
            .add_systems(PostUpdate, writeback_box3d_transforms);
    }
}

/// Settings for Ahoy's lightweight Box3D runtime.
#[derive(Clone, Copy, Debug, PartialEq, Resource)]
pub struct AhoyBox3dConfig {
    pub gravity: Vec3,
    pub sub_steps: i32,
}

impl Default for AhoyBox3dConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.8, 0.0),
            sub_steps: 4,
        }
    }
}

/// Rigid body mode for Ahoy's lightweight Box3D runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Box3dBodyType {
    Static,
    Kinematic,
    Dynamic,
}

impl From<Box3dBodyType> for box3d::BodyType {
    fn from(value: Box3dBodyType) -> Self {
        match value {
            Box3dBodyType::Static => Self::Static,
            Box3dBodyType::Kinematic => Self::Kinematic,
            Box3dBodyType::Dynamic => Self::Dynamic,
        }
    }
}

/// Body component for Ahoy's lightweight Box3D runtime.
#[derive(Clone, Copy, Debug, PartialEq, Component)]
pub struct AhoyBox3dBody {
    pub body_type: Box3dBodyType,
    pub linear_damping: f32,
    pub angular_damping: f32,
}

impl AhoyBox3dBody {
    pub const STATIC: Self = Self::new(Box3dBodyType::Static);
    pub const KINEMATIC: Self = Self::new(Box3dBodyType::Kinematic);
    pub const DYNAMIC: Self = Self::new(Box3dBodyType::Dynamic);

    pub const fn new(body_type: Box3dBodyType) -> Self {
        Self {
            body_type,
            linear_damping: 0.0,
            angular_damping: 0.0,
        }
    }
}

/// Collider shape for Ahoy's lightweight Box3D runtime.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Box3dColliderShape {
    Cuboid { half_extents: Vec3 },
    Sphere { radius: f32 },
}

/// Collider component for Ahoy's lightweight Box3D runtime.
#[derive(Clone, Copy, Debug, PartialEq, Component)]
pub struct AhoyBox3dCollider {
    pub shape: Box3dColliderShape,
    pub density: f32,
    pub friction: f32,
    pub sensor: bool,
}

impl AhoyBox3dCollider {
    pub const fn cuboid(half_extents: Vec3) -> Self {
        Self {
            shape: Box3dColliderShape::Cuboid { half_extents },
            density: 1.0,
            friction: 0.6,
            sensor: false,
        }
    }

    pub const fn sphere(radius: f32) -> Self {
        Self {
            shape: Box3dColliderShape::Sphere { radius },
            density: 1.0,
            friction: 0.6,
            sensor: false,
        }
    }
}

/// Linear and angular velocity synced to and from Box3D.
#[derive(Clone, Copy, Debug, Default, PartialEq, Component)]
pub struct AhoyBox3dVelocity {
    pub linear: Vec3,
    pub angular: Vec3,
}

impl AhoyBox3dVelocity {
    pub const fn linear(linear: Vec3) -> Self {
        Self {
            linear,
            angular: Vec3::ZERO,
        }
    }
}

/// Native Box3D body handle for an entity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Component)]
pub struct AhoyBox3dNativeBody {
    pub id: box3d::BodyId,
}

/// Native Box3D shape handle for an entity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Component)]
pub struct AhoyBox3dShape {
    pub id: box3d::ShapeId,
}

/// Non-send runtime resource containing the native Box3D world.
pub struct Box3dRuntime {
    world: box3d::World,
    bodies: HashMap<Entity, box3d::BodyId>,
    shapes: HashMap<Entity, box3d::ShapeId>,
    shape_bodies: HashMap<Entity, Entity>,
    body_entities: HashMap<u64, Entity>,
    shape_entities: HashMap<u64, Entity>,
}

impl Box3dRuntime {
    pub fn new(config: AhoyBox3dConfig) -> Self {
        Self {
            world: box3d::World::new(to_box3d_vec3(config.gravity)),
            bodies: HashMap::default(),
            shapes: HashMap::default(),
            shape_bodies: HashMap::default(),
            body_entities: HashMap::default(),
            shape_entities: HashMap::default(),
        }
    }

    pub fn world(&self) -> &box3d::World {
        &self.world
    }

    pub fn body(&self, entity: Entity) -> Option<box3d::BodyId> {
        self.bodies.get(&entity).copied()
    }

    pub fn shape(&self, entity: Entity) -> Option<box3d::ShapeId> {
        self.shapes.get(&entity).copied()
    }

    pub fn body_entity(&self, body: box3d::BodyId) -> Option<Entity> {
        self.body_entities.get(&body.to_bits()).copied()
    }

    pub fn shape_entity(&self, shape: box3d::ShapeId) -> Option<Entity> {
        self.shape_entities.get(&shape.to_bits()).copied()
    }

    fn remove_shape(&mut self, entity: Entity, destroy: bool) {
        if let Some(shape) = self.shapes.remove(&entity) {
            self.shape_entities.remove(&shape.to_bits());
            if destroy && shape.is_valid() {
                shape.destroy(true);
            }
        }
        self.shape_bodies.remove(&entity);
    }

    fn remove_body(&mut self, entity: Entity) {
        let shape_entities = self.shapes_for_body(entity);
        for shape_entity in shape_entities {
            self.remove_shape(shape_entity, false);
        }
        if let Some(body) = self.bodies.remove(&entity) {
            self.body_entities.remove(&body.to_bits());
            if body.is_valid() {
                body.destroy();
            }
        }
    }

    fn shapes_for_body(&self, body_entity: Entity) -> Vec<Entity> {
        self.shape_bodies
            .iter()
            .filter_map(|(shape_entity, owner)| (*owner == body_entity).then_some(*shape_entity))
            .collect()
    }
}

impl Drop for Box3dRuntime {
    fn drop(&mut self) {
        for (_, body) in self.bodies.drain() {
            if body.is_valid() {
                body.destroy();
            }
        }
    }
}

fn cleanup_box3d_shapes(
    mut runtime: NonSendMut<Box3dRuntime>,
    mut removed_shapes: RemovedComponents<AhoyBox3dShape>,
) {
    for entity in removed_shapes.read() {
        runtime.remove_shape(entity, true);
    }
}

fn cleanup_box3d_bodies(
    mut runtime: NonSendMut<Box3dRuntime>,
    mut removed_bodies: RemovedComponents<AhoyBox3dNativeBody>,
) {
    for entity in removed_bodies.read() {
        runtime.remove_body(entity);
    }
}

#[allow(clippy::type_complexity)]
fn create_box3d_bodies(
    mut commands: Commands,
    mut runtime: NonSendMut<Box3dRuntime>,
    bodies: Query<
        (
            Entity,
            &AhoyBox3dBody,
            Option<&Transform>,
            Option<&AhoyBox3dVelocity>,
        ),
        Without<AhoyBox3dNativeBody>,
    >,
) {
    for (entity, body, transform, velocity) in &bodies {
        let transform = transform.copied().unwrap_or_default();
        let native_body = runtime.world.create_body(box3d::BodyDef {
            body_type: body.body_type.into(),
            position: to_box3d_vec3(transform.translation),
            rotation: to_box3d_quat(transform.rotation),
            linear_velocity: velocity
                .map(|v| to_box3d_vec3(v.linear))
                .unwrap_or_default(),
            angular_velocity: velocity
                .map(|v| to_box3d_vec3(v.angular))
                .unwrap_or_default(),
            linear_damping: body.linear_damping,
            angular_damping: body.angular_damping,
            user_data: entity.to_bits() as usize,
            ..box3d::BodyDef::default()
        });
        let body_id = native_body.id();
        std::mem::forget(native_body);

        runtime.bodies.insert(entity, body_id);
        runtime.body_entities.insert(body_id.to_bits(), entity);
        commands
            .entity(entity)
            .insert(AhoyBox3dNativeBody { id: body_id });
    }
}

fn create_box3d_shapes(
    mut commands: Commands,
    mut runtime: NonSendMut<Box3dRuntime>,
    colliders: Query<(Entity, &AhoyBox3dCollider, &AhoyBox3dNativeBody), Without<AhoyBox3dShape>>,
) {
    for (entity, collider, body) in &colliders {
        let mut def = box3d::ShapeDef::default();
        def.density = collider.density;
        def.friction = collider.friction;
        def.is_sensor = collider.sensor;

        let body = body.id;
        let shape = match collider.shape {
            Box3dColliderShape::Cuboid { half_extents } => {
                body.create_box(to_box3d_vec3(half_extents), def)
            }
            Box3dColliderShape::Sphere { radius } => {
                body.create_sphere(box3d::Vec3::ZERO, radius, def)
            }
        };
        let shape_id = shape;

        runtime.shapes.insert(entity, shape_id);
        runtime.shape_bodies.insert(entity, entity);
        runtime.shape_entities.insert(shape_id.to_bits(), entity);
        commands
            .entity(entity)
            .insert(AhoyBox3dShape { id: shape_id });
    }
}

fn sync_box3d_body_changes(
    bodies: Query<
        (&AhoyBox3dBody, &Transform, &AhoyBox3dNativeBody),
        Or<(Changed<AhoyBox3dBody>, Changed<Transform>)>,
    >,
) {
    for (body, transform, native_body) in &bodies {
        native_body.id.set_body_type(body.body_type.into());
        native_body.id.set_transform(
            to_box3d_vec3(transform.translation),
            to_box3d_quat(transform.rotation),
        );
        native_body.id.set_linear_damping(body.linear_damping);
        native_body.id.set_angular_damping(body.angular_damping);
    }
}

fn sync_box3d_velocity_changes(
    velocities: Query<(&AhoyBox3dVelocity, &AhoyBox3dNativeBody), Changed<AhoyBox3dVelocity>>,
) {
    for (velocity, native_body) in &velocities {
        native_body
            .id
            .set_linear_velocity(to_box3d_vec3(velocity.linear));
        native_body
            .id
            .set_angular_velocity(to_box3d_vec3(velocity.angular));
    }
}

fn step_box3d(
    config: Res<AhoyBox3dConfig>,
    time: Res<bevy_time::Time>,
    runtime: NonSend<Box3dRuntime>,
) {
    let delta = time.delta_secs();
    if delta > 0.0 {
        runtime.world.step(delta, config.sub_steps);
    }
}

fn writeback_box3d_transforms(
    mut bodies: Query<(
        &mut Transform,
        Option<&mut AhoyBox3dVelocity>,
        &AhoyBox3dNativeBody,
    )>,
) {
    for (mut transform, velocity, native_body) in &mut bodies {
        let Some(native_transform) = native_body.id.transform() else {
            continue;
        };
        transform.translation = from_box3d_vec3(native_transform.p);
        transform.rotation = from_box3d_quat(native_transform.q);

        if let Some(mut velocity) = velocity {
            velocity.linear = from_box3d_vec3(native_body.id.linear_velocity());
            velocity.angular = from_box3d_vec3(native_body.id.angular_velocity());
        }
    }
}

fn to_box3d_vec3(value: Vec3) -> box3d::Vec3 {
    box3d::Vec3::new(value.x, value.y, value.z)
}

fn from_box3d_vec3(value: box3d::Vec3) -> Vec3 {
    Vec3::new(value.x, value.y, value.z)
}

fn to_box3d_quat(value: Quat) -> box3d::Quat {
    box3d::Quat::new(to_box3d_vec3(value.xyz()), value.w)
}

fn from_box3d_quat(value: box3d::Quat) -> Quat {
    Quat::from_xyzw(value.v.x, value.v.y, value.v.z, value.s)
}
