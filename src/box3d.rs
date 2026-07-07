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
use bevy_math::{Quat, Vec3, Vec3Swizzles};
use bevy_time::{Stopwatch, Time};
use bevy_transform::prelude::Transform;
use core::time::Duration;
use std::collections::HashMap;

use crate::{CharacterLook, input::AccumulatedInput, kcc};

pub use ::bevy_box3d;
pub use ::box3d;

/// Common Box3D imports for users experimenting with the `box3d` feature.
pub mod prelude {
    pub use super::{
        AhoyBox3dBody, AhoyBox3dCollider, AhoyBox3dConfig, AhoyBox3dPlugin, AhoyBox3dShape,
        AhoyBox3dVelocity, Box3dBodyType, Box3dCastHit, Box3dCharacterController,
        Box3dCharacterControllerState, Box3dColliderShape, Box3dRuntime,
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
                    run_box3d_kcc,
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

/// Source-style kinematic character controller backed by Box3D shape casts.
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[require(
    AccumulatedInput,
    Box3dCharacterControllerState,
    Transform,
    CharacterLook
)]
pub struct Box3dCharacterController {
    pub height: f32,
    pub radius: f32,
    pub ground_distance: f32,
    pub min_walk_cos: f32,
    pub stop_speed: f32,
    pub friction_hz: f32,
    pub acceleration_hz: f32,
    pub air_acceleration_hz: f32,
    pub gravity: f32,
    pub speed: f32,
    pub max_speed: f32,
    pub max_air_wish_speed: f32,
    pub jump_height: f32,
    pub coyote_time: Duration,
    pub jump_input_buffer: Duration,
    pub skin_width: f32,
    pub max_slides: usize,
}

impl Default for Box3dCharacterController {
    fn default() -> Self {
        Self {
            height: 1.8,
            radius: 0.7,
            ground_distance: 0.05,
            min_walk_cos: 40.0_f32.to_radians().cos(),
            stop_speed: 2.54,
            friction_hz: 12.0,
            acceleration_hz: 8.0,
            air_acceleration_hz: 12.0,
            gravity: 29.0,
            speed: 12.0,
            max_speed: 100.0,
            max_air_wish_speed: 0.76,
            jump_height: 1.8,
            coyote_time: Duration::from_millis(100),
            jump_input_buffer: Duration::from_millis(150),
            skin_width: 0.015,
            max_slides: 4,
        }
    }
}

/// Runtime state for [`Box3dCharacterController`].
#[derive(Clone, Debug, PartialEq, Component)]
pub struct Box3dCharacterControllerState {
    pub velocity: Vec3,
    pub grounded: Option<Box3dCastHit>,
    pub last_ground: Stopwatch,
}

impl Default for Box3dCharacterControllerState {
    fn default() -> Self {
        let mut last_ground = Stopwatch::new();
        last_ground.set_elapsed(Duration::MAX);
        Self {
            velocity: Vec3::ZERO,
            grounded: None,
            last_ground,
        }
    }
}

/// Hit data produced by Box3D character casts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box3dCastHit {
    pub entity: Option<Entity>,
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub collision_distance: f32,
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

fn step_box3d(config: Res<AhoyBox3dConfig>, time: Res<Time>, runtime: NonSend<Box3dRuntime>) {
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

fn run_box3d_kcc(
    runtime: NonSend<Box3dRuntime>,
    time: Res<Time>,
    mut characters: Query<(
        &Box3dCharacterController,
        &mut Box3dCharacterControllerState,
        &mut AccumulatedInput,
        &CharacterLook,
        &mut Transform,
    )>,
) {
    let delta = time.delta_secs();
    for (cfg, mut state, mut input, look, mut transform) in &mut characters {
        state.last_ground.tick(time.delta());
        update_box3d_grounded(&runtime, cfg, &mut state, &transform);

        if state.grounded.is_none() {
            state.velocity.y -= cfg.gravity * 0.5 * delta;
        }

        handle_box3d_jump(cfg, &mut state, &mut input);

        let wish_velocity = box3d_wish_velocity(cfg, &input, look);
        if state.grounded.is_some() {
            apply_box3d_friction(cfg, &mut state, delta);
            box3d_ground_accelerate(cfg, &mut state, wish_velocity, delta);
            state.velocity.y = state.velocity.y.min(0.0);
        } else {
            box3d_air_accelerate(cfg, &mut state, wish_velocity, delta);
        }

        validate_box3d_velocity(cfg, &mut state);
        let movement = state.velocity * delta;
        box3d_move_and_slide(&runtime, cfg, &mut state, &mut transform, movement);
        update_box3d_grounded(&runtime, cfg, &mut state, &transform);

        if state.grounded.is_some() {
            state.velocity.y = 0.0;
            state.last_ground.reset();
        } else {
            state.velocity.y -= cfg.gravity * 0.5 * delta;
        }

        validate_box3d_velocity(cfg, &mut state);
    }
}

fn handle_box3d_jump(
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    input: &mut AccumulatedInput,
) {
    if state.grounded.is_none() && state.last_ground.elapsed() > cfg.coyote_time {
        return;
    }
    let Some(jump_time) = input.jumped.clone() else {
        return;
    };
    if jump_time.elapsed() > cfg.jump_input_buffer {
        return;
    }

    input.jumped = None;
    state.grounded = None;
    state.last_ground.set_elapsed(cfg.coyote_time);
    state.velocity.y += (2.0 * cfg.gravity * cfg.jump_height).sqrt();
}

fn box3d_wish_velocity(
    cfg: &Box3dCharacterController,
    input: &AccumulatedInput,
    look: &CharacterLook,
) -> Vec3 {
    let movement = input.last_movement.unwrap_or_default();
    let mut forward = kcc::forward(look.to_quat());
    forward.y = 0.0;
    forward = forward.normalize_or_zero();
    let mut right = kcc::right(look.to_quat());
    right.y = 0.0;
    right = right.normalize_or_zero();

    let wish_velocity = movement.y * forward + movement.x * right;
    wish_velocity.normalize_or_zero() * cfg.speed
}

fn box3d_ground_accelerate(
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    wish_velocity: Vec3,
    delta: f32,
) {
    let Ok((wish_dir, wish_speed)) = bevy_math::Dir3::new_and_length(wish_velocity) else {
        return;
    };
    let current_speed = state.velocity.dot(*wish_dir);
    let add_speed = wish_speed - current_speed;
    if add_speed <= 0.0 {
        return;
    }
    let accel_speed = (wish_speed * cfg.acceleration_hz * delta).min(add_speed);
    state.velocity += accel_speed * wish_dir;
}

fn box3d_air_accelerate(
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    wish_velocity: Vec3,
    delta: f32,
) {
    let Ok((wish_dir, wish_speed)) = bevy_math::Dir3::new_and_length(wish_velocity) else {
        return;
    };
    let wish_speed_cap = wish_speed.min(cfg.max_air_wish_speed);
    let current_speed = state.velocity.dot(*wish_dir);
    let add_speed = wish_speed_cap - current_speed;
    if add_speed <= 0.0 {
        return;
    }
    let accel_speed = (wish_speed * cfg.air_acceleration_hz * delta).min(add_speed);
    state.velocity += accel_speed * wish_dir;
}

fn apply_box3d_friction(
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    delta: f32,
) {
    let speed = state.velocity.xz().length();
    if speed < 0.001 {
        return;
    }
    let control = speed.max(cfg.stop_speed);
    let drop = control * cfg.friction_hz * delta;
    let new_speed = (speed - drop).max(0.0);
    if new_speed != speed {
        state.velocity.x *= new_speed / speed;
        state.velocity.z *= new_speed / speed;
    }
}

fn box3d_move_and_slide(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    transform: &mut Transform,
    movement: Vec3,
) {
    let mut remaining = movement;
    for _ in 0..cfg.max_slides {
        if remaining.length_squared() <= f32::EPSILON {
            break;
        }

        let Some(hit) = cast_box3d_character(runtime, cfg, transform.translation, remaining) else {
            transform.translation += remaining;
            break;
        };

        let travel = hit.distance.max(0.0);
        let direction = remaining.normalize_or_zero();
        transform.translation += direction * travel;
        transform.translation += hit.normal * cfg.skin_width;

        let into_plane = state.velocity.dot(hit.normal).min(0.0);
        state.velocity -= into_plane * hit.normal;

        let traveled = direction * travel;
        remaining -= traveled;
        let blocked = remaining.dot(hit.normal).min(0.0);
        remaining -= blocked * hit.normal;
    }
}

fn update_box3d_grounded(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    transform: &Transform,
) {
    let hit = cast_box3d_character(
        runtime,
        cfg,
        transform.translation,
        Vec3::NEG_Y * cfg.ground_distance,
    );
    state.grounded = hit.filter(|hit| hit.normal.y >= cfg.min_walk_cos);
}

fn cast_box3d_character(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    origin: Vec3,
    movement: Vec3,
) -> Option<Box3dCastHit> {
    let distance = movement.length();
    if distance <= f32::EPSILON {
        return None;
    }

    let points = box3d_character_points(cfg);
    let proxy = box3d::ShapeProxy::new(&points, cfg.radius).ok()?;
    let mut closest: Option<Box3dCastHit> = None;
    runtime.world.cast_shape(
        to_box3d_vec3(origin),
        proxy,
        to_box3d_vec3(movement),
        box3d::QueryFilter::default(),
        |hit| {
            let collision_distance = hit.fraction * distance;
            let safe_distance = (collision_distance - cfg.skin_width).max(0.0);
            let shape = hit.shape.id();
            closest = Some(Box3dCastHit {
                entity: runtime.shape_entity(shape),
                distance: safe_distance,
                point: from_box3d_vec3(hit.point),
                normal: from_box3d_vec3(hit.normal).normalize_or_zero(),
                collision_distance,
            });
            hit.fraction
        },
    );
    closest
}

fn box3d_character_points(cfg: &Box3dCharacterController) -> [box3d::Vec3; 2] {
    let half_segment = (cfg.height * 0.5 - cfg.radius).max(0.0);
    [
        box3d::Vec3::new(0.0, -half_segment, 0.0),
        box3d::Vec3::new(0.0, half_segment, 0.0),
    ]
}

fn validate_box3d_velocity(
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
) {
    for value in state.velocity.as_mut() {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
    state.velocity = state.velocity.clamp_length_max(cfg.max_speed);
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
