//! Box3D interop and a lightweight Bevy 0.18 runtime.
//!
//! This module is enabled with the `box3d` feature and re-exports the
//! low-level [`box3d`] crate and the Bevy adapter [`bevy_box3d`].
//!
//! At the time this feature was added, `bevy_box3d` targets Bevy 0.19 while
//! Ahoy targets Bevy 0.18. [`AhoyBox3dPlugin`] provides a native Box3D runtime
//! and Source-style character controller that compile on Ahoy's Bevy version;
//! [`bevy_box3d`] is re-exported for projects that can use its Bevy 0.19
//! integration directly.

use bevy_app::{FixedUpdate, Plugin, PostUpdate};
use bevy_ecs::{
    prelude::*,
    system::{NonSend, NonSendMut},
};
use bevy_math::{Dir3, Quat, Vec3, Vec3Swizzles};
use bevy_time::{Stopwatch, Time};
use bevy_transform::prelude::Transform;
use core::time::Duration;
use std::collections::HashMap;

use crate::{
    CharacterControllerOutput, CharacterLook, TouchingEntity,
    input::AccumulatedInput,
    kcc,
    water::{WaterLevel, WaterState},
};

pub use ::bevy_box3d;
pub use ::box3d;

const MAX_BOX3D_DEPENETRATION_PLANES: usize = 8;

/// Common Box3D imports for users experimenting with the `box3d` feature.
pub mod prelude {
    pub use super::{
        AhoyBox3dBody, AhoyBox3dCollider, AhoyBox3dConfig, AhoyBox3dPlugin, AhoyBox3dShape,
        AhoyBox3dVelocity, Box3dBodyType, Box3dCastHit, Box3dCharacterController,
        Box3dCharacterControllerState, Box3dColliderShape, Box3dRuntime, Box3dWater,
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
                    spin_box3d_character_look,
                    apply_box3d_kcc_impulses,
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

/// Non-solid oriented cuboid water volume used by the Box3D controller.
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[require(Transform)]
pub struct Box3dWater {
    pub half_extents: Vec3,
    pub speed: f32,
}

impl Box3dWater {
    pub const fn cuboid(half_extents: Vec3) -> Self {
        Self {
            half_extents,
            speed: f32::MAX,
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
    CharacterControllerOutput,
    WaterState,
    Transform,
    CharacterLook
)]
pub struct Box3dCharacterController {
    pub height: f32,
    pub crouch_height: f32,
    pub radius: f32,
    pub view_height: f32,
    pub crouch_view_height: f32,
    pub crouch_speed_scale: f32,
    pub ground_distance: f32,
    pub min_walk_cos: f32,
    pub stop_speed: f32,
    pub friction_hz: f32,
    pub acceleration_hz: f32,
    pub air_acceleration_hz: f32,
    pub water_acceleration_hz: f32,
    pub water_slowdown: f32,
    pub water_gravity: f32,
    pub gravity: f32,
    pub speed: f32,
    pub max_speed: f32,
    pub max_air_wish_speed: f32,
    pub jump_height: f32,
    pub coyote_time: Duration,
    pub jump_input_buffer: Duration,
    pub skin_width: f32,
    pub max_slides: usize,
    pub push_mass: f32,
    pub step_size: f32,
    pub step_down_detection_distance: f32,
    pub min_step_ledge_space: f32,
}

impl Default for Box3dCharacterController {
    fn default() -> Self {
        Self {
            height: 1.8,
            crouch_height: 1.3,
            radius: 0.7,
            view_height: 1.7,
            crouch_view_height: 1.2,
            crouch_speed_scale: 1.0 / 3.0,
            ground_distance: 0.05,
            min_walk_cos: 40.0_f32.to_radians().cos(),
            stop_speed: 2.54,
            friction_hz: 12.0,
            acceleration_hz: 8.0,
            air_acceleration_hz: 12.0,
            water_acceleration_hz: 12.0,
            water_slowdown: 0.6,
            water_gravity: 2.4,
            gravity: 29.0,
            speed: 12.0,
            max_speed: 100.0,
            max_air_wish_speed: 0.76,
            jump_height: 1.8,
            coyote_time: Duration::from_millis(100),
            jump_input_buffer: Duration::from_millis(150),
            skin_width: 0.015,
            max_slides: 4,
            push_mass: 80.0,
            step_size: 0.7,
            step_down_detection_distance: 0.2,
            min_step_ledge_space: 0.2,
        }
    }
}

/// Runtime state for [`Box3dCharacterController`].
#[derive(Clone, Debug, PartialEq, Component)]
pub struct Box3dCharacterControllerState {
    pub velocity: Vec3,
    pub platform_velocity: Vec3,
    pub platform_angular_velocity: Vec3,
    pub grounded: Option<Box3dCastHit>,
    pub crouching: bool,
    pub last_ground: Stopwatch,
    pub last_step_up: Stopwatch,
    pub last_step_down: Stopwatch,
}

impl Default for Box3dCharacterControllerState {
    fn default() -> Self {
        let mut last_ground = Stopwatch::new();
        last_ground.set_elapsed(Duration::MAX);
        Self {
            velocity: Vec3::ZERO,
            platform_velocity: Vec3::ZERO,
            platform_angular_velocity: Vec3::ZERO,
            grounded: None,
            crouching: false,
            last_ground,
            last_step_up: max_stopwatch(),
            last_step_down: max_stopwatch(),
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

fn max_stopwatch() -> Stopwatch {
    let mut watch = Stopwatch::new();
    watch.set_elapsed(Duration::MAX);
    watch
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
    waters: Query<(&Box3dWater, &Transform), Without<Box3dCharacterController>>,
    mut characters: Query<(
        &Box3dCharacterController,
        &mut Box3dCharacterControllerState,
        &mut AccumulatedInput,
        &CharacterLook,
        &mut WaterState,
        &mut CharacterControllerOutput,
        &mut Transform,
    )>,
) {
    let delta = time.delta_secs();
    for (cfg, mut state, mut input, look, mut water, mut output, mut transform) in &mut characters {
        output.mantle = None;
        output.touching_entities.clear();
        state.last_ground.tick(time.delta());
        state.last_step_up.tick(time.delta());
        state.last_step_down.tick(time.delta());
        depenetrate_box3d_character(&runtime, cfg, &state, &mut transform);
        update_box3d_grounded(&runtime, cfg, &mut state, &transform, delta);
        handle_box3d_crouching(&runtime, cfg, &mut state, &input, &transform);
        update_box3d_water(cfg, &state, &transform, &waters, &mut water);
        if water.level > WaterLevel::Feet {
            state.grounded = None;
        }

        if state.grounded.is_none() && water.level <= WaterLevel::Feet {
            state.velocity.y -= cfg.gravity * 0.5 * delta;
        }

        handle_box3d_jump(cfg, &mut state, &mut input);

        let wish_velocity = box3d_wish_velocity(cfg, &state, &input, look);
        if water.level > WaterLevel::Feet {
            apply_box3d_water_friction(cfg, &mut state, delta);
            prepare_box3d_water_velocity(cfg, &mut state, &mut input, look, delta);
        } else if state.grounded.is_some() {
            apply_box3d_friction(&runtime, cfg, &mut state, delta);
            box3d_ground_accelerate(cfg, &mut state, wish_velocity, delta);
            state.velocity.y = state.velocity.y.min(0.0);
        } else {
            box3d_air_accelerate(cfg, &mut state, wish_velocity, delta);
        }

        validate_box3d_velocity(cfg, &mut state);
        if water.level > WaterLevel::Feet {
            box3d_water_move(
                &runtime,
                cfg,
                &mut state,
                &mut output,
                &mut transform,
                delta,
            );
        } else if state.grounded.is_some() {
            box3d_ground_move(
                &runtime,
                cfg,
                &mut state,
                &mut output,
                &mut transform,
                delta,
            );
        } else {
            let movement = state.velocity * delta;
            box3d_move_and_slide(
                &runtime,
                cfg,
                &mut state,
                &mut output,
                &mut transform,
                movement,
            );
        }
        update_box3d_grounded(&runtime, cfg, &mut state, &transform, delta);
        if water.level > WaterLevel::Feet {
            state.grounded = None;
        }

        if state.grounded.is_some() {
            state.velocity.y = 0.0;
            state.last_ground.reset();
        } else if water.level <= WaterLevel::Feet {
            state.velocity.y -= cfg.gravity * 0.5 * delta;
        }

        validate_box3d_velocity(cfg, &mut state);
    }
}

fn spin_box3d_character_look(
    mut characters: Query<(&Box3dCharacterControllerState, &mut CharacterLook)>,
    time: Res<Time>,
) {
    for (state, mut look) in &mut characters {
        if state.grounded.is_none() {
            continue;
        }
        *look = CharacterLook::from_quat(
            Quat::from_rotation_y(state.platform_angular_velocity.y * time.delta_secs())
                * look.to_quat(),
        );
    }
}

fn apply_box3d_kcc_impulses(
    runtime: NonSend<Box3dRuntime>,
    characters: Query<(&Box3dCharacterController, &CharacterControllerOutput)>,
) {
    for (cfg, output) in &characters {
        for touch in &output.touching_entities {
            let Some(body_entity) = runtime.shape_bodies.get(&touch.entity).copied() else {
                continue;
            };
            let Some(body) = runtime.body(body_entity) else {
                continue;
            };
            if !body.is_valid() || body.body_type() != box3d::BodyType::Dynamic {
                continue;
            }

            let touch_dir = -*touch.normal;
            let body_velocity =
                from_box3d_vec3(body.world_point_velocity(to_box3d_vec3(touch.point)));
            let relative_velocity = touch.character_velocity - body_velocity;
            let touch_velocity = touch_dir.dot(relative_velocity) * touch_dir;
            let impulse = touch_velocity * cfg.push_mass;
            if impulse.length_squared() <= f32::EPSILON || !impulse.is_finite() {
                continue;
            }

            body.apply_linear_impulse(to_box3d_vec3(impulse), to_box3d_vec3(touch.point), true);
        }
    }
}

fn box3d_ground_move(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    output: &mut CharacterControllerOutput,
    transform: &mut Transform,
    delta: f32,
) {
    state.velocity.y = 0.0;
    state.velocity += state.platform_velocity;
    let mut movement = state.velocity * delta;
    movement.y = 0.0;

    if movement.length_squared() < 0.0001 {
        state.velocity -= state.platform_velocity;
        snap_box3d_to_ground(runtime, cfg, state, transform);
        return;
    }

    if cast_box3d_character(runtime, cfg, state, transform.translation, movement).is_none() {
        transform.translation += movement;
        state.velocity -= state.platform_velocity;
        depenetrate_box3d_character(runtime, cfg, state, transform);
        snap_box3d_to_ground(runtime, cfg, state, transform);
        return;
    }

    box3d_step_move(runtime, cfg, state, output, transform, movement);
    state.velocity -= state.platform_velocity;
    snap_box3d_to_ground(runtime, cfg, state, transform);
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
    state.velocity.y += (2.0 * cfg.gravity * cfg.jump_height).sqrt() + state.platform_velocity.y;
}

fn box3d_wish_velocity(
    cfg: &Box3dCharacterController,
    state: &Box3dCharacterControllerState,
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
    let speed = if state.crouching {
        cfg.speed * cfg.crouch_speed_scale
    } else {
        cfg.speed
    };
    wish_velocity.normalize_or_zero() * speed
}

fn box3d_3d_wish_velocity(
    cfg: &Box3dCharacterController,
    state: &Box3dCharacterControllerState,
    input: &AccumulatedInput,
    look: &CharacterLook,
) -> Vec3 {
    let movement = input.last_movement.unwrap_or_default();
    let wish_velocity =
        movement.y * kcc::forward(look.to_quat()) + movement.x * kcc::right(look.to_quat());
    let speed = if state.crouching {
        cfg.speed * cfg.crouch_speed_scale
    } else {
        cfg.speed
    };
    wish_velocity.normalize_or_zero() * speed
}

fn prepare_box3d_water_velocity(
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    input: &mut AccumulatedInput,
    look: &CharacterLook,
    delta: f32,
) {
    let mut wish_velocity = box3d_3d_wish_velocity(cfg, state, input, look);
    if input.swim_up {
        input.swim_up = false;
        wish_velocity += Vec3::Y * cfg.speed;
    }
    wish_velocity = wish_velocity.clamp_length_max(cfg.speed);
    if wish_velocity == Vec3::ZERO {
        wish_velocity -= Vec3::Y * cfg.water_gravity;
    }
    wish_velocity *= cfg.water_slowdown;

    box3d_water_accelerate(cfg, state, wish_velocity, delta);
}

fn box3d_water_move(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    output: &mut CharacterControllerOutput,
    transform: &mut Transform,
    delta: f32,
) {
    state.velocity += state.platform_velocity;
    let movement = state.velocity * delta;
    box3d_move_and_slide(runtime, cfg, state, output, transform, movement);
    state.velocity -= state.platform_velocity;
}

fn box3d_water_accelerate(
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    wish_velocity: Vec3,
    delta: f32,
) {
    let Ok((wish_dir, wish_speed)) = Dir3::new_and_length(wish_velocity) else {
        return;
    };
    let current_speed = state.velocity.dot(*wish_dir);
    let add_speed = wish_speed - current_speed;
    if add_speed <= 0.0 {
        return;
    }
    let accel_speed = (wish_speed * cfg.water_acceleration_hz * delta).min(add_speed);
    state.velocity += accel_speed * wish_dir;
}

fn box3d_ground_accelerate(
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    wish_velocity: Vec3,
    delta: f32,
) {
    let Ok((wish_dir, wish_speed)) = Dir3::new_and_length(wish_velocity) else {
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
    let Ok((wish_dir, wish_speed)) = Dir3::new_and_length(wish_velocity) else {
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
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    delta: f32,
) {
    let speed = state.velocity.xz().length();
    if speed < 0.001 {
        return;
    }
    let control = speed.max(cfg.stop_speed);
    let surface_friction = state
        .grounded
        .and_then(|ground| ground.entity)
        .and_then(|entity| runtime.shape(entity))
        .filter(|shape| shape.is_valid())
        .map(|shape| shape.friction())
        .unwrap_or(1.0);
    let drop = control * cfg.friction_hz * surface_friction * delta;
    let new_speed = (speed - drop).max(0.0);
    if new_speed != speed {
        state.velocity.x *= new_speed / speed;
        state.velocity.z *= new_speed / speed;
    }
}

fn apply_box3d_water_friction(
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    delta: f32,
) {
    let speed = state.velocity.length();
    if speed < 0.001 {
        return;
    }
    let control = speed.max(cfg.stop_speed);
    let new_speed = (speed - control * cfg.friction_hz * delta).max(0.0);
    if new_speed != speed {
        state.velocity *= new_speed / speed;
    }
}

fn box3d_move_and_slide(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    output: &mut CharacterControllerOutput,
    transform: &mut Transform,
    movement: Vec3,
) {
    let mut remaining = movement;
    for _ in 0..cfg.max_slides {
        if remaining.length_squared() <= f32::EPSILON {
            break;
        }

        let Some(hit) = cast_box3d_character(runtime, cfg, state, transform.translation, remaining)
        else {
            transform.translation += remaining;
            break;
        };

        push_box3d_touch(output, &hit, transform.translation, state.velocity);

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

fn box3d_step_move(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    output: &mut CharacterControllerOutput,
    transform: &mut Transform,
    movement: Vec3,
) {
    let original_position = transform.translation;
    let original_velocity = state.velocity;
    let original_touching_entities = output.touching_entities.clone();

    box3d_move_and_slide(runtime, cfg, state, output, transform, movement);
    let down_touching_entities = output.touching_entities.clone();
    let down_position = transform.translation;
    let down_velocity = state.velocity;

    transform.translation = original_position;
    state.velocity = original_velocity;
    output.touching_entities = original_touching_entities;

    let up = Vec3::Y * cfg.step_size;
    let up_distance = cast_box3d_character(runtime, cfg, state, transform.translation, up)
        .map(|hit| hit.distance)
        .unwrap_or(cfg.step_size);
    transform.translation.y += up_distance;

    let forward_probe = state.velocity.normalize_or_zero() * cfg.min_step_ledge_space;
    if cast_box3d_character(runtime, cfg, state, transform.translation, forward_probe).is_some() {
        transform.translation = down_position;
        state.velocity = down_velocity;
        output.touching_entities = down_touching_entities;
        return;
    }

    box3d_move_and_slide(runtime, cfg, state, output, transform, movement);

    let down = Vec3::NEG_Y * cfg.step_size;
    let Some(down_hit) = cast_box3d_character(runtime, cfg, state, transform.translation, down)
    else {
        transform.translation = down_position;
        state.velocity = down_velocity;
        output.touching_entities = down_touching_entities;
        return;
    };
    if down_hit.normal.y < cfg.min_walk_cos {
        transform.translation = down_position;
        state.velocity = down_velocity;
        output.touching_entities = down_touching_entities;
        return;
    }

    transform.translation.y -= down_hit.distance;
    depenetrate_box3d_character(runtime, cfg, state, transform);
    let up_position = transform.translation;
    let down_dist = down_position.xz().distance_squared(original_position.xz());
    let up_dist = up_position.xz().distance_squared(original_position.xz());

    if down_dist >= up_dist {
        transform.translation = down_position;
        state.velocity = down_velocity;
        output.touching_entities = down_touching_entities;
    } else {
        state.velocity.y = down_velocity.y;
        state.last_step_up.reset();
    }
}

fn push_box3d_touch(
    output: &mut CharacterControllerOutput,
    hit: &Box3dCastHit,
    character_position: Vec3,
    character_velocity: Vec3,
) {
    let Some(entity) = hit.entity else {
        return;
    };
    let Ok(normal) = Dir3::new(hit.normal) else {
        return;
    };
    output.touching_entities.push(TouchingEntity {
        entity,
        distance: hit.distance,
        point: hit.point,
        normal,
        character_position,
        character_velocity,
        collision_distance: hit.collision_distance,
    });
}

fn snap_box3d_to_ground(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    transform: &mut Transform,
) {
    let up = Vec3::Y * cfg.ground_distance;
    let up_distance = cast_box3d_character(runtime, cfg, state, transform.translation, up)
        .map(|hit| hit.distance)
        .unwrap_or(cfg.ground_distance);

    let original_position = transform.translation;
    let start = transform.translation + Vec3::Y * up_distance;
    let down_distance = up_distance + cfg.step_size;

    let Some(hit) = cast_box3d_character(runtime, cfg, state, start, Vec3::NEG_Y * down_distance)
    else {
        return;
    };
    if hit.normal.y < cfg.min_walk_cos || hit.distance <= cfg.ground_distance {
        return;
    }

    transform.translation = start + Vec3::NEG_Y * hit.distance;
    if original_position.y - transform.translation.y > cfg.step_down_detection_distance {
        state.last_step_down.reset();
    }
    depenetrate_box3d_character(runtime, cfg, state, transform);
}

fn update_box3d_grounded(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    transform: &Transform,
    delta: f32,
) {
    let cast_distance = if state.platform_velocity.y < 0.0 {
        cfg.ground_distance - state.platform_velocity.y * delta
    } else {
        cfg.ground_distance
    };
    let hit = cast_box3d_character(
        runtime,
        cfg,
        state,
        transform.translation,
        Vec3::NEG_Y * cast_distance,
    );
    let old_ground = state.grounded;
    let new_ground = hit.filter(|hit| hit.normal.y >= cfg.min_walk_cos);

    if let Some(platform_hit) = new_ground.or(old_ground) {
        update_box3d_platform_velocity(runtime, state, platform_hit);
    }
    state.grounded = new_ground;
}

fn update_box3d_platform_velocity(
    runtime: &Box3dRuntime,
    state: &mut Box3dCharacterControllerState,
    ground: Box3dCastHit,
) {
    let Some(collider_entity) = ground.entity else {
        state.platform_velocity = Vec3::ZERO;
        state.platform_angular_velocity = Vec3::ZERO;
        return;
    };
    let Some(body_entity) = runtime.shape_bodies.get(&collider_entity).copied() else {
        state.platform_velocity = Vec3::ZERO;
        state.platform_angular_velocity = Vec3::ZERO;
        return;
    };
    let Some(body) = runtime.body(body_entity).filter(|body| body.is_valid()) else {
        state.platform_velocity = Vec3::ZERO;
        state.platform_angular_velocity = Vec3::ZERO;
        return;
    };

    state.platform_velocity =
        from_box3d_vec3(body.world_point_velocity(to_box3d_vec3(ground.point)));
    state.platform_angular_velocity = from_box3d_vec3(body.angular_velocity());
}

fn update_box3d_water(
    cfg: &Box3dCharacterController,
    state: &Box3dCharacterControllerState,
    transform: &Transform,
    waters: &Query<(&Box3dWater, &Transform), Without<Box3dCharacterController>>,
    water_state: &mut WaterState,
) {
    water_state.level = WaterLevel::None;
    water_state.speed = f32::MAX;

    let feet = transform.translation + Vec3::NEG_Y * (cfg.height * 0.5 - cfg.ground_distance);
    let active_height = if state.crouching {
        cfg.crouch_height.max(cfg.radius * 2.0)
    } else {
        cfg.height
    };
    let waist = feet + Vec3::Y * active_height * 0.5;
    let view_height = if state.crouching {
        cfg.crouch_view_height
    } else {
        cfg.view_height
    };
    let eye = feet + Vec3::Y * view_height;

    for (water, water_transform) in waters {
        let level = if box3d_water_contains(water, water_transform, eye) {
            WaterLevel::Head
        } else if box3d_water_contains(water, water_transform, waist) {
            WaterLevel::Waist
        } else if box3d_water_contains(water, water_transform, feet) {
            WaterLevel::Feet
        } else {
            WaterLevel::None
        };
        if level != WaterLevel::None {
            water_state.level = water_state.level.max(level);
            water_state.speed = water_state.speed.min(water.speed);
        }
    }
}

fn box3d_water_contains(water: &Box3dWater, transform: &Transform, point: Vec3) -> bool {
    let local_point = transform.compute_affine().inverse().transform_point3(point);
    local_point.abs().cmple(water.half_extents).all()
}

fn handle_box3d_crouching(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    input: &AccumulatedInput,
    transform: &Transform,
) {
    if input.crouched {
        state.crouching = true;
    } else if state.crouching
        && !box3d_character_intersects(runtime, cfg, transform.translation, false)
    {
        state.crouching = false;
    }
}

fn box3d_character_intersects(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    origin: Vec3,
    crouching: bool,
) -> bool {
    let points = box3d_character_points(cfg, crouching);
    let mut intersects = false;
    runtime.world.collide_mover(
        to_box3d_vec3(origin),
        points,
        cfg.radius,
        box3d::QueryFilter::default(),
        |_, _| {
            intersects = true;
            false
        },
    );
    intersects
}

fn depenetrate_box3d_character(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &Box3dCharacterControllerState,
    transform: &mut Transform,
) {
    let points = box3d_character_points(cfg, state.crouching);
    let mut planes = Vec::new();
    runtime.world.collide_mover(
        to_box3d_vec3(transform.translation),
        points,
        cfg.radius,
        box3d::QueryFilter::default(),
        |_, plane| {
            planes.push(box3d::CollisionPlane::rigid(plane.plane));
            planes.len() < MAX_BOX3D_DEPENETRATION_PLANES
        },
    );
    if planes.is_empty() {
        return;
    }

    let offset = box3d::solve_planes(box3d::Vec3::ZERO, &mut planes);
    transform.translation += from_box3d_vec3(offset);
}

fn cast_box3d_character(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &Box3dCharacterControllerState,
    origin: Vec3,
    movement: Vec3,
) -> Option<Box3dCastHit> {
    let distance = movement.length();
    if distance <= f32::EPSILON {
        return None;
    }

    let points = box3d_character_points(cfg, state.crouching);
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

fn box3d_character_points(cfg: &Box3dCharacterController, crouching: bool) -> [box3d::Vec3; 2] {
    let height = if crouching {
        cfg.crouch_height
    } else {
        cfg.height
    }
    .max(cfg.radius * 2.0);
    let center_offset = (height - cfg.height) * 0.5;
    let half_segment = (height * 0.5 - cfg.radius).max(0.0);
    [
        box3d::Vec3::new(0.0, center_offset - half_segment, 0.0),
        box3d::Vec3::new(0.0, center_offset + half_segment, 0.0),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crouching_keeps_capsule_feet_fixed() {
        let cfg = Box3dCharacterController::default();
        let standing = box3d_character_points(&cfg, false);
        let crouching = box3d_character_points(&cfg, true);

        let standing_foot = standing[0].y - cfg.radius;
        let crouching_foot = crouching[0].y - cfg.radius;
        assert_eq!(standing_foot, crouching_foot);
        assert!(crouching[1].y < standing[1].y);
    }

    #[test]
    fn crouch_height_cannot_make_capsule_shorter_than_its_diameter() {
        let cfg = Box3dCharacterController {
            crouch_height: 0.1,
            ..Default::default()
        };
        let crouching = box3d_character_points(&cfg, true);

        assert_eq!(crouching[0], crouching[1]);
        assert_eq!(crouching[0].y - cfg.radius, -cfg.height * 0.5);
    }

    #[test]
    fn standing_overlap_ignores_floor_contact_but_detects_low_ceiling() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let floor = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, -0.5, 0.0)));
        let _floor_shape =
            floor.create_box(box3d::Vec3::new(5.0, 0.5, 5.0), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let origin = Vec3::Y * cfg.height * 0.5;

        assert!(!box3d_character_intersects(&runtime, &cfg, origin, false));

        let ceiling = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, 1.75, 0.0)));
        let _ceiling_shape =
            ceiling.create_box(box3d::Vec3::new(2.0, 0.25, 2.0), box3d::ShapeDef::default());

        assert!(box3d_character_intersects(&runtime, &cfg, origin, false));
        assert!(!box3d_character_intersects(&runtime, &cfg, origin, true));
    }

    #[test]
    fn water_volume_respects_transform() {
        let water = Box3dWater::cuboid(Vec3::new(2.0, 1.0, 0.5));
        let transform = Transform::from_xyz(3.0, 2.0, -1.0)
            .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))
            .with_scale(Vec3::new(1.0, 2.0, 1.0));

        assert!(box3d_water_contains(
            &water,
            &transform,
            transform.transform_point(Vec3::new(1.5, 0.75, 0.25))
        ));
        assert!(!box3d_water_contains(
            &water,
            &transform,
            transform.transform_point(Vec3::new(2.1, 0.0, 0.0))
        ));
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
