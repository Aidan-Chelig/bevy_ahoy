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
    intern::Interned,
    prelude::*,
    schedule::ScheduleLabel,
    system::{NonSend, NonSendMut},
};
use bevy_math::{Dir3, Quat, Vec3, Vec3Swizzles};
use bevy_time::{Stopwatch, Time};
use bevy_transform::prelude::Transform;
use core::time::Duration;
use std::collections::{HashMap, HashSet};

use crate::{
    CharacterController, CharacterControllerOutput, CharacterLook, MantleOutput, TouchingEntity,
    input::AccumulatedInput,
    kcc,
    water::{WaterLevel, WaterState},
};

pub use ::bevy_box3d;
pub use ::box3d;

const MAX_BOX3D_DEPENETRATION_PLANES: usize = 8;
const MAX_BOX3D_CLIP_PASSES: usize = 32;

/// Common Box3D imports for users experimenting with the `box3d` feature.
pub mod prelude {
    pub use super::{
        AhoyBox3dBody, AhoyBox3dCollider, AhoyBox3dConfig, AhoyBox3dPlugin, AhoyBox3dShape,
        AhoyBox3dSystems, AhoyBox3dVelocity, Box3dBodyType, Box3dCastHit, Box3dCharacterController,
        Box3dCharacterControllerState, Box3dColliderShape, Box3dHolding, Box3dMantleState,
        Box3dPickupAction, Box3dPickupActor, Box3dPickupInput, Box3dRuntime, Box3dWater,
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
    pub schedule: Interned<dyn ScheduleLabel>,
}

impl Default for AhoyBox3dPlugin {
    fn default() -> Self {
        Self {
            config: AhoyBox3dConfig::default(),
            schedule: FixedUpdate.intern(),
        }
    }
}

impl AhoyBox3dPlugin {
    /// Create a Box3D runtime plugin that ticks in the given schedule.
    pub fn new(schedule: impl ScheduleLabel) -> Self {
        Self {
            schedule: schedule.intern(),
            ..Default::default()
        }
    }

    /// Create a Box3D runtime plugin with custom config in [`FixedUpdate`].
    pub fn with_config(config: AhoyBox3dConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }
}

impl Plugin for AhoyBox3dPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_message::<Box3dPickupInput>()
            .insert_resource(self.config)
            .insert_non_send_resource(Box3dRuntime::new(self.config))
            .configure_sets(self.schedule, AhoyBox3dSystems::Tick)
            .add_systems(
                self.schedule,
                (
                    cleanup_box3d_shapes,
                    cleanup_box3d_bodies,
                    create_box3d_bodies,
                    create_box3d_shapes,
                    sync_box3d_body_changes,
                    sync_box3d_velocity_changes,
                    handle_box3d_pickup_input,
                    update_box3d_pickup_holds,
                    step_box3d,
                    run_box3d_kcc,
                    spin_box3d_character_look,
                    apply_box3d_kcc_impulses,
                )
                    .chain()
                    .in_set(AhoyBox3dSystems::Tick),
            )
            .add_systems(PostUpdate, writeback_box3d_transforms);
    }
}

#[derive(SystemSet, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum AhoyBox3dSystems {
    Tick,
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
    Cuboid {
        half_extents: Vec3,
    },
    CuboidAt {
        half_extents: Vec3,
        translation: Vec3,
        rotation: Quat,
    },
    Sphere {
        radius: f32,
    },
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

    pub fn cuboid_from_size(size: Vec3) -> Self {
        Self::cuboid(size * 0.5)
    }

    pub const fn cuboid_at(half_extents: Vec3, translation: Vec3, rotation: Quat) -> Self {
        Self {
            shape: Box3dColliderShape::CuboidAt {
                half_extents,
                translation,
                rotation,
            },
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

    pub const fn with_density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }

    pub const fn with_friction(mut self, friction: f32) -> Self {
        self.friction = friction;
        self
    }

    pub const fn sensor(mut self) -> Self {
        self.sensor = true;
        self
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

/// Source-style pickup actor for dynamic Box3D bodies.
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[require(Transform, CharacterLook)]
pub struct Box3dPickupActor {
    pub prop_filter: box3d::QueryFilter,
    pub max_distance: f32,
    pub preferred_distance: f32,
    pub hold_hz: f32,
    pub throw_speed: f32,
    pub max_prop_mass: f32,
}

impl Default for Box3dPickupActor {
    fn default() -> Self {
        Self {
            prop_filter: box3d::QueryFilter::default(),
            max_distance: 3.0,
            preferred_distance: 1.0,
            hold_hz: 14.0,
            throw_speed: 12.0,
            max_prop_mass: 1000.0,
        }
    }
}

/// Body currently held by a [`Box3dPickupActor`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Component)]
pub struct Box3dHolding {
    pub body: Entity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Box3dPickupAction {
    Pull,
    Drop,
    Throw,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Message)]
pub struct Box3dPickupInput {
    pub action: Box3dPickupAction,
    pub actor: Entity,
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
    pub air_speed: f32,
    pub max_speed: f32,
    pub max_air_wish_speed: f32,
    pub jump_height: f32,
    pub air_friction: f32,
    pub tac_power: f32,
    pub tac_jump_factor: f32,
    pub tac_input_buffer: Duration,
    pub max_tac_cos: f32,
    pub tac_cooldown: Duration,
    pub mantle_input_buffer: Duration,
    pub climbdown_input_buffer: Duration,
    pub mantle_height: f32,
    pub mantle_speed: f32,
    pub min_mantle_cos: f32,
    pub min_mantle_ledge_space: f32,
    pub min_ledge_grab_space: Vec3,
    pub max_ledge_grab_distance: f32,
    pub climb_pull_up_height: f32,
    pub climb_reverse_sin: f32,
    pub climb_sensitivity: f32,
    pub ledge_jump_power: f32,
    pub ledge_jump_factor: f32,
    pub crane_input_buffer: Duration,
    pub crane_height: f32,
    pub crane_speed: f32,
    pub min_crane_cos: f32,
    pub min_crane_ledge_space: f32,
    pub jump_crane_chain_time: Duration,
    pub unground_speed: f32,
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
        Self::from(CharacterController::default())
    }
}

impl From<CharacterController> for Box3dCharacterController {
    fn from(value: CharacterController) -> Self {
        Self {
            height: 1.8,
            crouch_height: value.crouch_height,
            radius: 0.7,
            view_height: value.standing_view_height,
            crouch_view_height: value.crouch_view_height,
            crouch_speed_scale: value.crouch_speed_scale,
            ground_distance: value.ground_distance,
            min_walk_cos: value.min_walk_cos,
            stop_speed: value.stop_speed,
            friction_hz: value.friction_hz,
            acceleration_hz: value.acceleration_hz,
            air_acceleration_hz: value.air_acceleration_hz,
            water_acceleration_hz: value.water_acceleration_hz,
            water_slowdown: value.water_slowdown,
            water_gravity: value.water_gravity,
            gravity: value.gravity,
            speed: value.speed,
            air_speed: value.air_speed,
            max_speed: value.max_speed,
            max_air_wish_speed: value.max_air_wish_speed,
            jump_height: value.jump_height,
            air_friction: 0.0,
            tac_power: value.tac_power,
            tac_jump_factor: value.tac_jump_factor,
            tac_input_buffer: value.tac_input_buffer,
            max_tac_cos: value.max_tac_cos,
            tac_cooldown: value.tac_cooldown,
            mantle_input_buffer: value.mantle_input_buffer,
            climbdown_input_buffer: value.climbdown_input_buffer,
            mantle_height: value.mantle_height,
            mantle_speed: value.mantle_speed,
            min_mantle_cos: value.min_mantle_cos,
            min_mantle_ledge_space: value.min_mantle_ledge_space,
            min_ledge_grab_space: value.min_ledge_grab_space.half_size * 2.0,
            max_ledge_grab_distance: value.max_ledge_grab_distance,
            climb_pull_up_height: value.climb_pull_up_height,
            climb_reverse_sin: value.climb_reverse_sin,
            climb_sensitivity: value.climb_sensitivity,
            ledge_jump_power: value.ledge_jump_power,
            ledge_jump_factor: value.ledge_jump_factor,
            crane_input_buffer: value.crane_input_buffer,
            crane_height: value.crane_height,
            crane_speed: value.crane_speed,
            min_crane_cos: value.min_crane_cos,
            min_crane_ledge_space: value.min_crane_ledge_space,
            jump_crane_chain_time: value.jump_crane_chain_time,
            unground_speed: value.unground_speed,
            coyote_time: value.coyote_time,
            jump_input_buffer: value.jump_input_buffer,
            skin_width: value.move_and_slide.skin_width,
            max_slides: value.move_and_slide.max_planes,
            push_mass: 80.0,
            step_size: value.step_size,
            step_down_detection_distance: value.step_down_detection_distance,
            min_step_ledge_space: value.min_step_ledge_space,
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
    pub tac_velocity: f32,
    pub crane_height_left: Option<f32>,
    pub mantle: Option<Box3dMantleState>,
    pub last_ground: Stopwatch,
    pub last_tac: Stopwatch,
    pub last_step_up: Stopwatch,
    pub last_step_down: Stopwatch,
    pub held_body: Option<Entity>,
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
            tac_velocity: 0.0,
            crane_height_left: None,
            mantle: None,
            last_ground,
            last_tac: max_stopwatch(),
            last_step_up: max_stopwatch(),
            last_step_down: max_stopwatch(),
            held_body: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box3dMantleState {
    pub wall_normal: Dir3,
    pub ledge_position: Vec3,
    pub wall_entity: Entity,
    pub target_position: Vec3,
    pub speed: f32,
    pub automatic: bool,
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
        let mut shapes: Vec<_> = self
            .shape_bodies
            .iter()
            .filter_map(|(shape_entity, owner)| (*owner == body_entity).then_some(*shape_entity))
            .collect();
        shapes.sort_by_key(|entity| entity.to_bits());
        shapes
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
    let mut removed: Vec<_> = removed_shapes.read().collect();
    removed.sort_by_key(|entity| entity.to_bits());
    for entity in removed {
        runtime.remove_shape(entity, true);
    }
}

fn cleanup_box3d_bodies(
    mut runtime: NonSendMut<Box3dRuntime>,
    mut removed_bodies: RemovedComponents<AhoyBox3dNativeBody>,
) {
    let mut removed: Vec<_> = removed_bodies.read().collect();
    removed.sort_by_key(|entity| entity.to_bits());
    for entity in removed {
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
    let mut bodies: Vec<_> = bodies.iter().collect();
    bodies.sort_by_key(|(entity, ..)| entity.to_bits());
    for (entity, body, transform, velocity) in bodies {
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
    let mut colliders: Vec<_> = colliders.iter().collect();
    colliders.sort_by_key(|(entity, ..)| entity.to_bits());
    for (entity, collider, body) in colliders {
        let mut def = box3d::ShapeDef::default();
        def.density = collider.density;
        def.friction = collider.friction;
        def.is_sensor = collider.sensor;

        let body = body.id;
        let shape = match collider.shape {
            Box3dColliderShape::Cuboid { half_extents } => {
                body.create_box(to_box3d_vec3(half_extents), def)
            }
            Box3dColliderShape::CuboidAt {
                half_extents,
                translation,
                rotation,
            } => body.create_transformed_box(
                to_box3d_vec3(half_extents),
                box3d::Transform {
                    p: to_box3d_vec3(translation),
                    q: to_box3d_quat(rotation),
                },
                def,
            ),
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

fn handle_box3d_pickup_input(
    mut commands: Commands,
    runtime: NonSend<Box3dRuntime>,
    mut inputs: MessageReader<Box3dPickupInput>,
    actors: Query<(
        &Box3dPickupActor,
        &Transform,
        &CharacterLook,
        Option<&Box3dHolding>,
    )>,
    mut character_states: Query<&mut Box3dCharacterControllerState>,
) {
    for input in inputs.read() {
        match input.action {
            Box3dPickupAction::Pull => {
                let Ok((actor, transform, look, holding)) = actors.get(input.actor) else {
                    continue;
                };
                if holding.is_some() {
                    continue;
                }
                let Some(body_entity) =
                    find_box3d_pickup_body(&runtime, actor, transform.translation, look)
                else {
                    continue;
                };
                commands
                    .entity(input.actor)
                    .insert(Box3dHolding { body: body_entity });
                if let Ok(mut state) = character_states.get_mut(input.actor) {
                    state.held_body = Some(body_entity);
                }
            }
            Box3dPickupAction::Drop | Box3dPickupAction::Throw => {
                let Ok((actor, _transform, look, holding)) = actors.get(input.actor) else {
                    continue;
                };
                let Some(holding) = holding else {
                    continue;
                };
                if input.action == Box3dPickupAction::Throw
                    && let Some(body) = runtime.body(holding.body)
                    && body.is_valid()
                {
                    body.set_linear_velocity(to_box3d_vec3(
                        box3d_pickup_forward(look) * actor.throw_speed,
                    ));
                }
                commands.entity(input.actor).remove::<Box3dHolding>();
                if let Ok(mut state) = character_states.get_mut(input.actor) {
                    state.held_body = None;
                }
            }
        }
    }
}

fn update_box3d_pickup_holds(
    runtime: NonSend<Box3dRuntime>,
    time: Res<Time>,
    actors: Query<(&Box3dPickupActor, &Transform, &CharacterLook, &Box3dHolding)>,
) {
    let delta = time.delta_secs();
    if delta <= 0.0 {
        return;
    }
    for (actor, transform, look, holding) in &actors {
        let Some(body) = runtime.body(holding.body) else {
            continue;
        };
        if !body.is_valid() || body.body_type() != box3d::BodyType::Dynamic {
            continue;
        }
        let target =
            transform.translation + box3d_pickup_forward(look) * actor.preferred_distance.max(0.0);
        let position = from_box3d_vec3(body.world_center_of_mass());
        let desired_velocity = (target - position) * actor.hold_hz.max(0.0);
        body.set_linear_velocity(to_box3d_vec3(desired_velocity));
        body.set_angular_velocity(box3d::Vec3::ZERO);
    }
}

fn find_box3d_pickup_body(
    runtime: &Box3dRuntime,
    actor: &Box3dPickupActor,
    origin: Vec3,
    look: &CharacterLook,
) -> Option<Entity> {
    let mut hit_body = None;
    runtime.world.cast_ray(
        to_box3d_vec3(origin),
        to_box3d_vec3(box3d_pickup_forward(look) * actor.max_distance.max(0.0)),
        actor.prop_filter,
        |hit| {
            let Some(shape_entity) = runtime.shape_entity(hit.shape.id()) else {
                return 1.0;
            };
            let Some(body_entity) = runtime.shape_bodies.get(&shape_entity).copied() else {
                return 1.0;
            };
            let Some(body) = runtime.body(body_entity) else {
                return 1.0;
            };
            if !body.is_valid() || body.body_type() != box3d::BodyType::Dynamic {
                return hit.fraction;
            }
            let inverse_mass = body.inverse_mass();
            if inverse_mass <= 0.0 {
                return hit.fraction;
            }
            let mass = inverse_mass.recip();
            if mass > actor.max_prop_mass {
                return hit.fraction;
            }
            hit_body = Some(body_entity);
            hit.fraction
        },
    );
    hit_body
}

fn box3d_pickup_forward(look: &CharacterLook) -> Vec3 {
    look.to_quat() * Vec3::NEG_Z
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
        prepare_box3d_kcc_tick(&time, &mut state, &mut output);

        depenetrate_box3d_character(&runtime, cfg, &state, &mut transform);
        update_box3d_grounded(&runtime, cfg, &mut state, &transform, delta);

        handle_box3d_crouching(&runtime, cfg, &mut state, &input, &transform);
        update_box3d_water(cfg, &state, &transform, &waters, &mut water);
        if water.level > WaterLevel::Feet {
            state.grounded = None;
        }

        if water.level <= WaterLevel::Feet && state.grounded.is_none() {
            state.velocity.y -= cfg.gravity * 0.5 * delta;
        }

        let wish_velocity = calculate_wish_velocity(cfg, &state, &input, look);
        let wish_velocity_3d = calculate_3d_wish_velocity(cfg, &state, &input, look);
        update_box3d_crane_state(
            &runtime,
            cfg,
            &mut state,
            &mut input,
            &transform,
            wish_velocity,
        );
        if state.crane_height_left.is_some() {
            state.mantle = None;
        } else {
            update_box3d_mantle_state(
                &runtime,
                cfg,
                &mut state,
                &mut input,
                &transform,
                wish_velocity,
            );
        }
        handle_box3d_ledge_jump(cfg, &mut state, &mut input, look);

        if state.crane_height_left.is_some() {
            handle_box3d_crane_movement(
                &runtime,
                cfg,
                &mut state,
                &mut output,
                &mut transform,
                wish_velocity,
                delta,
            );
        } else if state.mantle.is_some() {
            handle_box3d_jump(
                &runtime,
                cfg,
                &mut state,
                &mut input,
                &transform,
                wish_velocity,
                delta,
            );
            handle_box3d_mantle_movement(
                &runtime,
                cfg,
                &mut state,
                &input,
                look,
                &mut output,
                &mut transform,
                wish_velocity_3d,
                delta,
            );
        } else {
            handle_box3d_jump(
                &runtime,
                cfg,
                &mut state,
                &mut input,
                &transform,
                wish_velocity,
                delta,
            );
            apply_box3d_friction_for_state(&runtime, cfg, &mut state, &water, delta);
            validate_box3d_velocity(cfg, &mut state);
            move_box3d_character_for_state(
                &runtime,
                cfg,
                &mut state,
                &mut input,
                &mut output,
                &mut transform,
                &water,
                wish_velocity,
                look,
                delta,
            );
        }

        let was_grounded = state.grounded.is_some();
        update_box3d_grounded(&runtime, cfg, &mut state, &transform, delta);
        if water.level > WaterLevel::Feet {
            state.grounded = None;
        }
        if was_grounded {
            update_box3d_climbdown_state(
                &runtime,
                cfg,
                &mut state,
                &mut input,
                &mut transform,
                wish_velocity,
            );
        }

        finish_box3d_kcc_tick(cfg, &water, &mut state, delta);
        validate_box3d_velocity(cfg, &mut state);
    }
}

fn prepare_box3d_kcc_tick(
    time: &Time,
    state: &mut Box3dCharacterControllerState,
    output: &mut CharacterControllerOutput,
) {
    output.mantle = None;
    output.touching_entities.clear();
    state.last_ground.tick(time.delta());
    state.last_tac.tick(time.delta());
    state.last_step_up.tick(time.delta());
    state.last_step_down.tick(time.delta());
    state.tac_velocity *= 0.99;
}

fn apply_box3d_friction_for_state(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    water: &WaterState,
    delta: f32,
) {
    if water.level > WaterLevel::Feet {
        apply_box3d_water_friction(cfg, state, delta);
    } else if state.grounded.is_some() {
        apply_box3d_friction(runtime, cfg, state, delta);
    } else {
        apply_box3d_air_friction(cfg, state, delta);
    }
}

#[allow(clippy::too_many_arguments)]
fn move_box3d_character_for_state(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    input: &mut AccumulatedInput,
    output: &mut CharacterControllerOutput,
    transform: &mut Transform,
    water: &WaterState,
    wish_velocity: Vec3,
    look: &CharacterLook,
    delta: f32,
) {
    if water.level > WaterLevel::Feet {
        prepare_box3d_water_velocity(cfg, state, input, look, delta);
        box3d_water_move(runtime, cfg, state, output, transform, delta);
    } else if state.grounded.is_some() {
        ground_accelerate(cfg, state, wish_velocity, delta);
        state.velocity.y = state.velocity.y.min(0.0);
        box3d_ground_move(runtime, cfg, state, output, transform, delta);
    } else {
        air_accelerate(cfg, state, wish_velocity, delta);
        box3d_move_and_slide(
            runtime,
            cfg,
            state,
            output,
            transform,
            state.velocity * delta,
        );
    }
}

fn finish_box3d_kcc_tick(
    cfg: &Box3dCharacterController,
    water: &WaterState,
    state: &mut Box3dCharacterControllerState,
    delta: f32,
) {
    if state.grounded.is_some() {
        state.velocity.y = state.platform_velocity.y;
        state.last_ground.reset();
    } else if water.level <= WaterLevel::Feet {
        state.velocity.y -= cfg.gravity * 0.5 * delta;
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
    characters: Query<(
        Entity,
        &Box3dCharacterController,
        &CharacterControllerOutput,
    )>,
) {
    let mut impulses = Vec::new();
    for (character_entity, cfg, output) in &characters {
        let mut pushed_bodies = HashSet::new();
        for touch in &output.touching_entities {
            let Some(body_entity) = runtime.shape_bodies.get(&touch.entity).copied() else {
                continue;
            };
            if !pushed_bodies.insert(body_entity) {
                continue;
            }
            let Some(body) = runtime.body(body_entity) else {
                continue;
            };
            if !body.is_valid() || body.body_type() != box3d::BodyType::Dynamic {
                continue;
            }

            let impulse = calculate_box3d_push_impulse(
                body,
                cfg.push_mass,
                touch.point,
                -*touch.normal,
                touch.character_velocity,
            );
            if impulse.length_squared() <= f32::EPSILON || !impulse.is_finite() {
                continue;
            }

            impulses.push((body_entity, character_entity, touch.point, impulse));
        }
    }
    impulses.sort_by_key(|(body_entity, character_entity, ..)| {
        (body_entity.to_bits(), character_entity.to_bits())
    });
    for (body_entity, _, point, impulse) in impulses {
        let Some(body) = runtime.body(body_entity) else {
            continue;
        };
        if body.is_valid() {
            body.apply_linear_impulse(to_box3d_vec3(impulse), to_box3d_vec3(point), true);
        }
    }
}

fn calculate_box3d_push_impulse(
    body: box3d::BodyId,
    character_mass: f32,
    point: Vec3,
    direction: Vec3,
    character_velocity: Vec3,
) -> Vec3 {
    if character_mass <= 0.0 || !character_mass.is_finite() {
        return Vec3::ZERO;
    }
    let direction = direction.normalize_or_zero();
    let body_velocity = from_box3d_vec3(body.world_point_velocity(to_box3d_vec3(point)));
    let closing_speed = direction.dot(character_velocity - body_velocity).max(0.0);
    if closing_speed <= f32::EPSILON {
        return Vec3::ZERO;
    }

    let lever = point - from_box3d_vec3(body.world_center_of_mass());
    let angular_axis = lever.cross(direction);
    let inverse_inertia = body.world_inverse_rotational_inertia();
    let angular_response = from_box3d_vec3(inverse_inertia.cx) * angular_axis.x
        + from_box3d_vec3(inverse_inertia.cy) * angular_axis.y
        + from_box3d_vec3(inverse_inertia.cz) * angular_axis.z;
    let angular_inverse_mass = direction.dot(angular_response.cross(lever)).max(0.0);
    let inverse_effective_mass =
        character_mass.recip() + body.inverse_mass() + angular_inverse_mass;
    if inverse_effective_mass <= f32::EPSILON || !inverse_effective_mass.is_finite() {
        return Vec3::ZERO;
    }

    direction * (closing_speed / inverse_effective_mass)
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

fn update_box3d_crane_state(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    input: &mut AccumulatedInput,
    transform: &Transform,
    wish_velocity: Vec3,
) {
    if state.mantle.is_some() {
        return;
    }
    if state.crane_height_left.is_some() {
        state.mantle = None;
        return;
    }
    let Some(crane_time) = input.craned.clone() else {
        return;
    };
    if crane_time.elapsed() > cfg.crane_input_buffer {
        return;
    }

    let Some(crane_height) =
        available_box3d_crane_height(runtime, cfg, state, transform, wish_velocity)
    else {
        state.crane_height_left = None;
        return;
    };

    input.craned = None;
    input.jumped = None;
    input.mantled = None;
    input.tac = None;
    state.mantle = None;
    state.grounded = None;
    state.crane_height_left = Some(crane_height);
}

fn available_box3d_crane_height(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &Box3dCharacterControllerState,
    transform: &Transform,
    wish_velocity: Vec3,
) -> Option<f32> {
    let wish_dir = Dir3::new(wish_velocity)
        .or_else(|_| Dir3::new(Vec3::new(state.velocity.x, 0.0, state.velocity.z)))
        .ok()?;
    let wall_hit = cast_box3d_character(
        runtime,
        cfg,
        state,
        transform.translation,
        *wish_dir * cfg.min_crane_ledge_space,
    )?;
    let wall_normal = Vec3::new(wall_hit.normal.x, 0.0, wall_hit.normal.z).normalize_or_zero();
    if (-wall_normal).dot(*wish_dir) < cfg.min_crane_cos {
        return None;
    }

    let up_hit = cast_box3d_character(
        runtime,
        cfg,
        state,
        transform.translation,
        Vec3::Y * cfg.crane_height,
    );
    let up_dist = up_hit.map(|hit| hit.distance).unwrap_or(cfg.crane_height);
    let probe_origin =
        transform.translation + Vec3::Y * up_dist - wall_normal * cfg.min_crane_ledge_space;
    let down_hit = cast_box3d_character(runtime, cfg, state, probe_origin, Vec3::NEG_Y * up_dist)?;
    if down_hit.normal.y < cfg.min_walk_cos {
        return None;
    }
    let crane_height = up_dist - down_hit.distance;
    if crane_height <= cfg.step_size || crane_height > cfg.crane_height {
        return None;
    }

    let landing_position =
        probe_origin + Vec3::NEG_Y * down_hit.distance + Vec3::Y * cfg.skin_width;
    if box3d_character_intersects(runtime, cfg, landing_position, state.crouching) {
        return None;
    }
    Some(crane_height)
}

fn handle_box3d_crane_movement(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    output: &mut CharacterControllerOutput,
    transform: &mut Transform,
    wish_velocity: Vec3,
    delta: f32,
) {
    let Some(crane_height) = state.crane_height_left else {
        return;
    };
    state.velocity.y = 0.0;
    ground_accelerate(cfg, state, wish_velocity, delta);
    state.velocity.y = 0.0;
    state.velocity += state.platform_velocity;

    let Ok((vel_dir, speed)) = Dir3::new_and_length(state.velocity) else {
        state.crane_height_left = None;
        state.velocity -= state.platform_velocity;
        return;
    };
    let wish_dir = Dir3::new(wish_velocity).unwrap_or(vel_dir);
    state.velocity -= state.platform_velocity;

    let Some(wall_hit) = cast_box3d_character(
        runtime,
        cfg,
        state,
        transform.translation,
        *wish_dir * cfg.min_crane_ledge_space,
    ) else {
        state.crane_height_left = None;
        return;
    };
    let wall_normal = Vec3::new(wall_hit.normal.x, 0.0, wall_hit.normal.z).normalize_or_zero();
    if (-wall_normal).dot(*wish_dir) < cfg.min_crane_cos {
        state.crane_height_left = None;
        return;
    }

    let vertical = Vec3::Y * (cfg.crane_speed * delta).min(crane_height);
    let top_hit = cast_box3d_character(runtime, cfg, state, transform.translation, vertical);
    let travel_dist = top_hit.map(|hit| hit.distance).unwrap_or(vertical.y);
    transform.translation.y += travel_dist;

    let saved_velocity = state.velocity;
    state.velocity = state.platform_velocity;
    box3d_move_and_slide(
        runtime,
        cfg,
        state,
        output,
        transform,
        state.velocity * delta,
    );
    state.velocity = saved_velocity;

    state.crane_height_left = if top_hit.is_some() {
        Some(0.0)
    } else {
        Some((crane_height - travel_dist).max(0.0))
    };
    state.last_step_up.reset();

    if state.crane_height_left != Some(0.0) {
        if cast_box3d_character(
            runtime,
            cfg,
            state,
            transform.translation,
            *vel_dir * cfg.min_crane_ledge_space,
        )
        .is_none()
        {
            transform.translation += *vel_dir * speed * delta;
            depenetrate_box3d_character(runtime, cfg, state, transform);
            state.crane_height_left = None;
        }
        return;
    }

    if cast_box3d_character(
        runtime,
        cfg,
        state,
        transform.translation,
        *vel_dir * cfg.min_crane_ledge_space,
    )
    .is_some()
    {
        state.crane_height_left = None;
        return;
    }
    transform.translation += *vel_dir * speed * delta;
    depenetrate_box3d_character(runtime, cfg, state, transform);
    state.crane_height_left = None;
}

fn update_box3d_mantle_state(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    input: &mut AccumulatedInput,
    transform: &Transform,
    wish_velocity: Vec3,
) {
    if state.mantle.is_some() {
        return;
    }
    let Some(mantle_time) = input.mantled.as_ref() else {
        return;
    };
    if mantle_time.elapsed() > cfg.mantle_input_buffer {
        return;
    }
    let Ok(wish_dir) = Dir3::new(wish_velocity) else {
        return;
    };
    let Some(wall_hit) = cast_box3d_character(
        runtime,
        cfg,
        state,
        transform.translation,
        *wish_dir * cfg.max_ledge_grab_distance,
    ) else {
        return;
    };
    let Ok(wall_normal) = Dir3::new(wall_hit.normal) else {
        return;
    };
    if (-wall_normal).dot(*wish_dir) < cfg.min_mantle_cos {
        return;
    }
    let Some(wall_entity) = wall_hit.entity else {
        return;
    };

    let hand_radius = (cfg.min_ledge_grab_space.x.max(cfg.min_ledge_grab_space.z) * 0.5).max(0.01);
    let inward_distance = cfg.radius + cfg.skin_width + cfg.min_ledge_grab_space.z * 0.5;
    let probe_start = transform.translation + Vec3::Y * (cfg.height * 0.5 + cfg.mantle_height)
        - *wall_normal * inward_distance;
    let probe_distance = cfg.height + cfg.mantle_height;
    let Some(ledge_hit) = cast_box3d_sphere(
        runtime,
        probe_start,
        Vec3::NEG_Y * probe_distance,
        hand_radius,
        cfg.skin_width,
    ) else {
        return;
    };
    if ledge_hit.normal.y < cfg.min_walk_cos {
        return;
    }

    let feet_y = transform.translation.y - cfg.height * 0.5;
    let ledge_height = ledge_hit.point.y - feet_y;
    if ledge_height <= cfg.step_size || ledge_height > cfg.height + cfg.mantle_height {
        return;
    }
    let target_position = Vec3::new(
        ledge_hit.point.x,
        ledge_hit.point.y + cfg.height * 0.5 + cfg.skin_width + cfg.climb_pull_up_height,
        ledge_hit.point.z,
    );
    if box3d_character_intersects(runtime, cfg, target_position, state.crouching) {
        return;
    }

    state.grounded = None;
    state.velocity = Vec3::ZERO;
    state.mantle = Some(Box3dMantleState {
        wall_normal,
        ledge_position: ledge_hit.point,
        wall_entity,
        target_position,
        speed: cfg.mantle_speed,
        automatic: false,
    });
    input.craned = None;
    input.mantled = None;
    input.jumped = None;
}

fn update_box3d_climbdown_state(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    input: &mut AccumulatedInput,
    transform: &mut Transform,
    wish_velocity: Vec3,
) {
    if state.grounded.is_some() || state.mantle.is_some() {
        return;
    }
    if input.last_movement.unwrap_or_default().y >= 0.0 {
        return;
    }
    let Some(climbdown_time) = input.climbdown.clone() else {
        return;
    };
    if climbdown_time.elapsed() > cfg.climbdown_input_buffer {
        return;
    }
    if cast_box3d_character(
        runtime,
        cfg,
        state,
        transform.translation,
        Vec3::NEG_Y * cfg.crane_height,
    )
    .is_some()
    {
        return;
    }

    let original_position = transform.translation;
    let saved_mantle_input = input.mantled.take();
    transform.translation += Vec3::NEG_Y * cfg.crane_height;
    input.mantled = Some(climbdown_time);
    update_box3d_mantle_state(runtime, cfg, state, input, transform, -wish_velocity);
    transform.translation = original_position;

    if state.mantle.is_some() {
        input.craned = None;
        input.mantled = None;
        input.jumped = None;
        input.climbdown = None;
        input.tac = None;
    } else {
        input.mantled = saved_mantle_input;
    }
}

fn handle_box3d_mantle_movement(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    input: &AccumulatedInput,
    look: &CharacterLook,
    output: &mut CharacterControllerOutput,
    transform: &mut Transform,
    wish_velocity: Vec3,
    delta: f32,
) {
    let Some(mantle) = state.mantle else {
        return;
    };
    output.mantle = Some(MantleOutput {
        wall_normal: mantle.wall_normal,
        ledge_position: mantle.ledge_position,
        wall_entity: mantle.wall_entity,
    });
    state.velocity = Vec3::ZERO;

    let to_target = mantle.target_position - transform.translation;
    if to_target.length_squared() <= cfg.skin_width * cfg.skin_width {
        transform.translation = mantle.target_position;
        state.mantle = None;
        state.last_step_up.reset();
        return;
    }

    let climb_factor = if mantle.automatic {
        1.0
    } else {
        calculate_box3d_climb_factor(cfg, input, look, wish_velocity)
    };
    if !mantle.automatic && climb_factor.abs() <= f32::EPSILON {
        return;
    }

    let vertical_left = to_target.y.max(0.0);
    let desired = if vertical_left > cfg.skin_width {
        Vec3::Y * (mantle.speed * delta * climb_factor).clamp(-vertical_left, vertical_left)
    } else {
        to_target
    };
    let movement = desired.clamp_length_max(mantle.speed * delta);
    let travel = cast_box3d_character(runtime, cfg, state, transform.translation, movement)
        .map(|hit| movement.normalize_or_zero() * hit.distance)
        .unwrap_or(movement);
    transform.translation += travel;
    state.velocity = travel / delta;
    state.last_step_up.reset();

    if travel.length_squared() + f32::EPSILON < movement.length_squared() {
        state.mantle = None;
    }
}

fn calculate_box3d_climb_factor(
    cfg: &Box3dCharacterController,
    input: &AccumulatedInput,
    look: &CharacterLook,
    wish_velocity: Vec3,
) -> f32 {
    if wish_velocity.length_squared() < 0.01 {
        return 0.0;
    }
    let movement = input.last_movement.unwrap_or_default().y;
    let cos = (kcc::forward(look.to_quat()) * movement.abs()).y;
    let factor = ((cos + cfg.climb_reverse_sin) * cfg.climb_sensitivity).clamp(-1.0, 1.0);
    if movement < 0.0 { -factor } else { factor }
}

fn handle_box3d_ledge_jump(
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    input: &mut AccumulatedInput,
    look: &CharacterLook,
) {
    let Some(mantle_time) = input.mantled.as_ref() else {
        return;
    };
    if state.mantle.is_none()
        || mantle_time.elapsed() < cfg.mantle_input_buffer
        || input.jumped.is_none()
    {
        return;
    }
    let movement = input.last_movement.unwrap_or_default();
    let direction = if movement.y >= 0.0 {
        let forward = kcc::forward(look.to_quat());
        let flat_forward = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
        (Vec3::Y * cfg.ledge_jump_factor + flat_forward).normalize_or_zero()
    } else {
        Vec3::NEG_Y
    };

    state.mantle = None;
    state.last_tac.reset();
    input.jumped = None;
    input.mantled = None;
    input.tac = None;
    state.velocity +=
        direction * cfg.ledge_jump_power * (2.0 * cfg.gravity * cfg.jump_height).sqrt();
}

fn handle_box3d_jump(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    input: &mut AccumulatedInput,
    transform: &Transform,
    wish_velocity: Vec3,
    delta: f32,
) {
    let jump_direction =
        if state.grounded.is_none() && state.last_ground.elapsed() > cfg.coyote_time {
            let Some(direction) =
                handle_box3d_tac(runtime, cfg, state, input, transform, wish_velocity, delta)
            else {
                return;
            };
            direction
        } else {
            let Some(jump_time) = input.jumped.clone() else {
                return;
            };
            if jump_time.elapsed() > cfg.jump_input_buffer {
                return;
            }
            state.grounded = None;
            state.last_ground.set_elapsed(cfg.coyote_time);
            Vec3::Y
        };

    input.jumped = None;
    input.tac = None;
    state.last_tac.reset();
    state.velocity += jump_direction * (2.0 * cfg.gravity * cfg.jump_height).sqrt()
        + Vec3::Y * state.platform_velocity.y;
    if let Some(crane_input) = input.craned.as_mut() {
        crane_input.tick((cfg.crane_input_buffer - cfg.jump_crane_chain_time).max(Duration::ZERO));
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_box3d_tac(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    input: &AccumulatedInput,
    transform: &Transform,
    wish_velocity: Vec3,
    delta: f32,
) -> Option<Vec3> {
    let tac_time = input.tac.as_ref()?;
    if tac_time.elapsed() > cfg.tac_input_buffer
        || wish_velocity.length_squared() < 0.1
        || state.last_tac.elapsed() < cfg.tac_cooldown
    {
        return None;
    }

    let normal = cast_box3d_character(
        runtime,
        cfg,
        state,
        transform.translation,
        state.velocity * delta,
    )
    .or_else(|| {
        cast_box3d_character(
            runtime,
            cfg,
            state,
            transform.translation,
            wish_velocity * delta,
        )
    })?
    .normal;

    calculate_box3d_tac_direction(cfg, state, normal, wish_velocity)
}

fn calculate_box3d_tac_direction(
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    normal: Vec3,
    wish_velocity: Vec3,
) -> Option<Vec3> {
    if normal.y < -0.01 {
        return None;
    }

    let wish_unit = wish_velocity.normalize();
    let wish_dot = wish_unit.dot(normal);
    if -wish_dot > cfg.max_tac_cos {
        return None;
    }

    let vel_dot = state.velocity.dot(normal).min(0.0);
    state.velocity -= vel_dot * normal;
    let groundedness = state.tac_velocity.max(vel_dot).min(1.0);
    state.tac_velocity = 0.0;
    let flat_normal = Vec3::new(normal.x, 0.0, normal.z);
    let tac_wish = wish_unit - (wish_dot.min(0.0) - 1.0) * flat_normal;
    let tac_direction = (Vec3::Y * cfg.tac_jump_factor + tac_wish).normalize_or_zero();
    Some(tac_direction * groundedness * cfg.tac_power)
}

fn calculate_wish_velocity(
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
    let speed = if state.grounded.is_none() {
        cfg.air_speed
    } else if state.crouching {
        cfg.speed * cfg.crouch_speed_scale
    } else {
        cfg.speed
    };
    wish_velocity.normalize_or_zero() * speed
}

fn calculate_3d_wish_velocity(
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
    let mut wish_velocity = calculate_3d_wish_velocity(cfg, state, input, look);
    if input.swim_up {
        input.swim_up = false;
        wish_velocity += Vec3::Y * cfg.speed;
    }
    wish_velocity = wish_velocity.clamp_length_max(cfg.speed);
    if wish_velocity == Vec3::ZERO {
        wish_velocity -= Vec3::Y * cfg.water_gravity;
    }
    wish_velocity *= cfg.water_slowdown;

    water_accelerate(cfg, state, wish_velocity, delta);
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

fn water_accelerate(
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

fn ground_accelerate(
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

fn air_accelerate(
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

fn apply_box3d_air_friction(
    cfg: &Box3dCharacterController,
    state: &mut Box3dCharacterControllerState,
    delta: f32,
) {
    if cfg.air_friction <= 0.0 {
        return;
    }
    let speed = state.velocity.length();
    if speed < 0.001 {
        return;
    }
    let control = speed.max(cfg.stop_speed);
    let new_speed = (speed - control * cfg.friction_hz * cfg.air_friction * delta).max(0.0);
    if new_speed != speed {
        state.velocity *= new_speed / speed;
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
    let mut planes = Vec::with_capacity(cfg.max_slides);
    for _ in 0..cfg.max_slides {
        if remaining.length_squared() <= f32::EPSILON {
            break;
        }

        if add_box3d_overlap_planes(
            runtime,
            cfg,
            state,
            transform.translation,
            remaining,
            &mut planes,
        ) {
            let old_velocity = state.velocity;
            state.velocity = clip_box3d_kcc_vector(state.velocity, &planes);
            state.tac_velocity += (old_velocity - state.velocity).length();
            remaining = clip_box3d_kcc_vector(remaining, &planes);
            continue;
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
        if travel > 1.0e-4 && hit.normal.y >= -0.01 {
            transform.translation += hit.normal * cfg.skin_width;
        }

        if !planes.iter().any(|plane: &box3d::CollisionPlane| {
            from_box3d_vec3(plane.plane().normal).dot(hit.normal) > 0.999
        }) {
            planes.push(box3d::CollisionPlane::rigid(box3d::Plane {
                normal: to_box3d_vec3(hit.normal),
                offset: 0.0,
            }));
        }
        let old_velocity = state.velocity;
        state.velocity = clip_box3d_kcc_vector(state.velocity, &planes);
        state.tac_velocity += (old_velocity - state.velocity).length();

        let traveled = direction * travel;
        remaining -= traveled;
        remaining = clip_box3d_kcc_vector(remaining, &planes);
    }
}

fn clip_box3d_kcc_vector(mut vector: Vec3, planes: &[box3d::CollisionPlane]) -> Vec3 {
    let original_y = vector.y;
    for _ in 0..MAX_BOX3D_CLIP_PASSES {
        let before = vector;
        for plane in planes {
            let normal = from_box3d_vec3(plane.plane().normal).normalize_or_zero();
            let into = vector.dot(normal);
            if into < 0.0 {
                vector -= normal * into;
            }
        }
        if vector.distance_squared(before) <= 1.0e-10 {
            break;
        }
    }
    if original_y <= 0.0 && vector.y < original_y {
        vector.y = original_y;
    }
    if vector.length_squared() <= 1.0e-10 {
        vector = Vec3::ZERO;
    }
    vector
}

fn add_box3d_overlap_planes(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &Box3dCharacterControllerState,
    origin: Vec3,
    movement: Vec3,
    planes: &mut Vec<box3d::CollisionPlane>,
) -> bool {
    let points = box3d_character_points(cfg, state.crouching);
    let mut added = false;
    runtime.world.collide_mover(
        to_box3d_vec3(origin),
        points,
        cfg.radius,
        box3d::QueryFilter::default(),
        |shape, plane| {
            if box3d_state_holds_shape(runtime, state, shape.id()) {
                return true;
            }
            let normal = from_box3d_vec3(plane.plane.normal).normalize_or_zero();
            if movement.dot(normal) < -1.0e-4
                && !planes
                    .iter()
                    .any(|existing| from_box3d_vec3(existing.plane().normal).dot(normal) > 0.999)
            {
                planes.push(box3d::CollisionPlane::rigid(box3d::Plane {
                    normal: to_box3d_vec3(normal),
                    offset: 0.0,
                }));
                added = true;
            }
            planes.len() < MAX_BOX3D_DEPENETRATION_PLANES
        },
    );
    added
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
    let original_tac_velocity = state.tac_velocity;
    let original_touching_entities = output.touching_entities.clone();

    box3d_move_and_slide(runtime, cfg, state, output, transform, movement);
    let down_touching_entities = output.touching_entities.clone();
    let down_position = transform.translation;
    let down_velocity = state.velocity;
    let down_tac_velocity = state.tac_velocity;

    transform.translation = original_position;
    state.velocity = original_velocity;
    state.tac_velocity = original_tac_velocity;
    output.touching_entities = original_touching_entities;

    let up = Vec3::Y * cfg.step_size;
    let up_hit = cast_box3d_character(runtime, cfg, state, transform.translation, up);
    if up_hit.is_some_and(|hit| hit.normal.y < -0.01 && hit.distance < cfg.step_size) {
        transform.translation = down_position;
        state.velocity = down_velocity;
        state.tac_velocity = down_tac_velocity;
        output.touching_entities = down_touching_entities;
        return;
    }
    let up_distance = up_hit.map(|hit| hit.distance).unwrap_or(cfg.step_size);
    transform.translation.y += up_distance;

    let forward_probe = state.velocity.normalize_or_zero() * cfg.min_step_ledge_space;
    if cast_box3d_character(runtime, cfg, state, transform.translation, forward_probe).is_some() {
        transform.translation = down_position;
        state.velocity = down_velocity;
        state.tac_velocity = down_tac_velocity;
        output.touching_entities = down_touching_entities;
        return;
    }

    box3d_move_and_slide(runtime, cfg, state, output, transform, movement);

    let down = Vec3::NEG_Y * cfg.step_size;
    let Some(down_hit) = cast_box3d_character(runtime, cfg, state, transform.translation, down)
    else {
        transform.translation = down_position;
        state.velocity = down_velocity;
        state.tac_velocity = down_tac_velocity;
        output.touching_entities = down_touching_entities;
        return;
    };
    if down_hit.normal.y < cfg.min_walk_cos {
        transform.translation = down_position;
        state.velocity = down_velocity;
        state.tac_velocity = down_tac_velocity;
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
        state.tac_velocity = down_tac_velocity;
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
    let mut moving_up_rapidly = state.velocity.y > cfg.unground_speed;
    if moving_up_rapidly && state.grounded.is_some() {
        moving_up_rapidly = (state.velocity.y - state.platform_velocity.y) > cfg.unground_speed;
    }
    if moving_up_rapidly {
        state.grounded = None;
        return;
    }

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
    )
    .or_else(|| overlap_box3d_ground(runtime, cfg, state, transform.translation));
    let old_ground = state.grounded;
    let new_ground = hit.filter(|hit| hit.normal.y >= cfg.min_walk_cos);

    if let Some(platform_hit) = new_ground.or(old_ground) {
        update_box3d_platform_velocity(runtime, state, platform_hit);
    }
    state.grounded = new_ground;
}

fn overlap_box3d_ground(
    runtime: &Box3dRuntime,
    cfg: &Box3dCharacterController,
    state: &Box3dCharacterControllerState,
    origin: Vec3,
) -> Option<Box3dCastHit> {
    let points = box3d_character_points(cfg, state.crouching);
    let mut ground = None;
    runtime.world.collide_mover(
        to_box3d_vec3(origin),
        points,
        cfg.radius,
        box3d::QueryFilter::default(),
        |shape, plane| {
            if box3d_state_holds_shape(runtime, state, shape.id()) {
                return true;
            }
            let normal = from_box3d_vec3(plane.plane.normal).normalize_or_zero();
            if normal.y >= cfg.min_walk_cos {
                ground = Some(Box3dCastHit {
                    entity: runtime.shape_entity(shape.id()),
                    distance: 0.0,
                    point: from_box3d_vec3(plane.point),
                    normal,
                    collision_distance: 0.0,
                });
                return false;
            }
            true
        },
    );
    ground
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
        |_, plane| {
            let normal = from_box3d_vec3(plane.plane.normal).normalize_or_zero();
            if normal.y >= cfg.min_walk_cos {
                return true;
            }
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
        |shape, plane| {
            if box3d_state_holds_shape(runtime, state, shape.id()) {
                return true;
            }
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
            if box3d_state_holds_shape(runtime, state, hit.shape.id()) {
                return 1.0;
            }
            let collision_distance = hit.fraction * distance;
            let normal = from_box3d_vec3(hit.normal).normalize_or_zero();
            if movement.dot(normal) >= -1.0e-4 {
                return closest
                    .map(|hit| hit.collision_distance / distance)
                    .unwrap_or(1.0);
            }
            let safe_distance = (collision_distance - cfg.skin_width).max(0.0);
            let shape = hit.shape.id();
            closest = Some(Box3dCastHit {
                entity: runtime.shape_entity(shape),
                distance: safe_distance,
                point: from_box3d_vec3(hit.point),
                normal,
                collision_distance,
            });
            hit.fraction
        },
    );
    if closest.is_none() && movement.y > 0.0 {
        let direction = movement / distance;
        let cast_movement = movement + direction * cfg.skin_width;
        let cast_distance = cast_movement.length();
        runtime.world.cast_shape(
            to_box3d_vec3(origin - direction * cfg.skin_width),
            proxy,
            to_box3d_vec3(cast_movement),
            box3d::QueryFilter::default(),
            |hit| {
                if box3d_state_holds_shape(runtime, state, hit.shape.id()) {
                    return 1.0;
                }
                let collision_distance = hit.fraction * cast_distance;
                let normal = from_box3d_vec3(hit.normal).normalize_or_zero();
                if movement.dot(normal) >= -1.0e-4 {
                    return closest
                        .map(|hit| hit.collision_distance / cast_distance)
                        .unwrap_or(1.0);
                }
                let safe_distance = (collision_distance - cfg.skin_width).max(0.0);
                let shape = hit.shape.id();
                closest = Some(Box3dCastHit {
                    entity: runtime.shape_entity(shape),
                    distance: safe_distance.min(distance),
                    point: from_box3d_vec3(hit.point),
                    normal,
                    collision_distance,
                });
                hit.fraction
            },
        );
    }
    closest
}

fn box3d_state_holds_shape(
    runtime: &Box3dRuntime,
    state: &Box3dCharacterControllerState,
    shape: box3d::ShapeId,
) -> bool {
    let Some(held_body) = state.held_body else {
        return false;
    };
    let Some(shape_entity) = runtime.shape_entity(shape) else {
        return false;
    };
    runtime.shape_bodies.get(&shape_entity) == Some(&held_body)
}

fn cast_box3d_sphere(
    runtime: &Box3dRuntime,
    origin: Vec3,
    movement: Vec3,
    radius: f32,
    skin_width: f32,
) -> Option<Box3dCastHit> {
    let distance = movement.length();
    if distance <= f32::EPSILON {
        return None;
    }
    let points = [box3d::Vec3::ZERO];
    let proxy = box3d::ShapeProxy::new(&points, radius).ok()?;
    let mut closest = None;
    runtime.world.cast_shape(
        to_box3d_vec3(origin),
        proxy,
        to_box3d_vec3(movement),
        box3d::QueryFilter::default(),
        |hit| {
            let collision_distance = hit.fraction * distance;
            let safe_distance = (collision_distance - skin_width).max(0.0);
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
    fn box3d_defaults_track_avian_controller_defaults() {
        let avian = CharacterController::default();
        let box3d = Box3dCharacterController::default();

        assert_eq!(box3d.crouch_height, avian.crouch_height);
        assert_eq!(box3d.view_height, avian.standing_view_height);
        assert_eq!(box3d.speed, avian.speed);
        assert_eq!(box3d.air_speed, avian.air_speed);
        assert_eq!(box3d.jump_height, avian.jump_height);
        assert_eq!(box3d.crane_height, avian.crane_height);
        assert_eq!(box3d.mantle_height, avian.mantle_height);
        assert_eq!(box3d.skin_width, avian.move_and_slide.skin_width);
        assert_eq!(box3d.max_slides, avian.move_and_slide.max_planes);
    }

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
    fn standing_probe_ignores_tiny_floor_overlap() {
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
        let origin = Vec3::Y * (cfg.height * 0.5 - cfg.skin_width);

        assert!(!box3d_character_intersects(&runtime, &cfg, origin, false));
    }

    #[test]
    fn crouch_release_stands_after_leaving_low_space() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let ceiling = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(-1.0, 1.55, 0.0)));
        let _ceiling_shape =
            ceiling.create_box(box3d::Vec3::new(1.0, 0.1, 1.0), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            crouching: true,
            ..Default::default()
        };
        let input = AccumulatedInput::default();
        let mut transform = Transform::from_xyz(-1.0, cfg.height * 0.5, 0.0);

        handle_box3d_crouching(&runtime, &cfg, &mut state, &input, &transform);
        assert!(state.crouching);

        transform.translation.x = 1.0;
        handle_box3d_crouching(&runtime, &cfg, &mut state, &input, &transform);
        assert!(!state.crouching);
    }

    #[test]
    fn crouch_release_stands_after_leaving_sloped_low_space() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let ceiling = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::ZERO));
        let _ceiling_shape = ceiling.id().create_transformed_box(
            box3d::Vec3::new(1.1, 0.08, 1.0),
            box3d::Transform {
                p: box3d::Vec3::new(-1.0, 1.58, 0.0),
                q: to_box3d_quat(Quat::from_rotation_z(18.0_f32.to_radians())),
            },
            box3d::ShapeDef::default(),
        );
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            crouching: true,
            ..Default::default()
        };
        let input = AccumulatedInput::default();
        let mut transform = Transform::from_xyz(-1.0, cfg.height * 0.5, 0.0);

        handle_box3d_crouching(&runtime, &cfg, &mut state, &input, &transform);
        assert!(state.crouching);

        transform.translation.x = 1.0;
        handle_box3d_crouching(&runtime, &cfg, &mut state, &input, &transform);
        assert!(!state.crouching);
    }

    #[test]
    fn held_body_is_ignored_by_character_casts() {
        let mut runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let body_entity = Entity::from_bits(42);
        let shape_entity = body_entity;
        let wall = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, 0.9, -1.0)));
        let shape = wall.create_box(box3d::Vec3::new(1.0, 0.9, 0.1), box3d::ShapeDef::default());
        let body_id = wall.id();
        let shape_id = shape.id();
        runtime.bodies.insert(body_entity, body_id);
        runtime.shapes.insert(shape_entity, shape_id);
        runtime.shape_bodies.insert(shape_entity, body_entity);
        runtime
            .shape_entities
            .insert(shape_id.to_bits(), shape_entity);

        let cfg = Box3dCharacterController::default();
        let clear_state = Box3dCharacterControllerState::default();
        let held_state = Box3dCharacterControllerState {
            held_body: Some(body_entity),
            ..Default::default()
        };
        let origin = Vec3::Y * cfg.height * 0.5;

        assert!(
            cast_box3d_character(&runtime, &cfg, &clear_state, origin, Vec3::NEG_Z * 2.0).is_some()
        );
        assert!(
            cast_box3d_character(&runtime, &cfg, &held_state, origin, Vec3::NEG_Z * 2.0).is_none()
        );
    }

    #[test]
    fn pickup_ray_stops_at_static_body() {
        let mut runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let wall_entity = Entity::from_bits(41);
        let prop_entity = Entity::from_bits(42);
        let wall = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, 0.0, -1.0)));
        let wall_shape =
            wall.create_box(box3d::Vec3::new(0.5, 0.5, 0.05), box3d::ShapeDef::default());
        let prop = runtime
            .world
            .create_body(box3d::BodyDef::dynamic_at(box3d::Vec3::new(0.0, 0.0, -2.0)));
        let prop_shape = prop.create_box(
            box3d::Vec3::new(0.25, 0.25, 0.25),
            box3d::ShapeDef {
                density: 1.0,
                ..Default::default()
            },
        );
        runtime.bodies.insert(wall_entity, wall.id());
        runtime.shapes.insert(wall_entity, wall_shape.id());
        runtime.shape_bodies.insert(wall_entity, wall_entity);
        runtime
            .shape_entities
            .insert(wall_shape.id().to_bits(), wall_entity);
        runtime.bodies.insert(prop_entity, prop.id());
        runtime.shapes.insert(prop_entity, prop_shape.id());
        runtime.shape_bodies.insert(prop_entity, prop_entity);
        runtime
            .shape_entities
            .insert(prop_shape.id().to_bits(), prop_entity);

        let picked = find_box3d_pickup_body(
            &runtime,
            &Box3dPickupActor::default(),
            Vec3::ZERO,
            &CharacterLook::default(),
        );

        assert_eq!(picked, None);
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

    #[test]
    fn tac_redirects_glancing_movement_away_from_wall() {
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            tac_velocity: 1.0,
            ..Default::default()
        };
        let direction = calculate_box3d_tac_direction(
            &cfg,
            &mut state,
            Vec3::NEG_X,
            Vec3::new(0.5, 0.0, 0.866),
        )
        .unwrap();

        assert!(direction.x < 0.0);
        assert!(direction.y > 0.0);
        assert!(direction.z > 0.0);
        assert!((direction.length() - cfg.tac_power).abs() < 0.0001);
        assert_eq!(state.tac_velocity, 0.0);
    }

    #[test]
    fn tac_rejects_head_on_wall_approach() {
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            tac_velocity: 1.0,
            ..Default::default()
        };

        assert!(calculate_box3d_tac_direction(&cfg, &mut state, Vec3::NEG_X, Vec3::X).is_none());
        assert_eq!(state.tac_velocity, 1.0);
    }

    #[test]
    fn airborne_wish_velocity_uses_air_speed() {
        let cfg = Box3dCharacterController::default();
        let state = Box3dCharacterControllerState::default();
        let input = AccumulatedInput {
            last_movement: Some(bevy_math::Vec2::Y),
            ..Default::default()
        };

        let wish_velocity =
            calculate_wish_velocity(&cfg, &state, &input, &CharacterLook::default());

        assert!((wish_velocity.length() - cfg.air_speed).abs() < 0.0001);
    }

    #[test]
    fn air_friction_damps_airborne_velocity() {
        let cfg = Box3dCharacterController {
            air_friction: 1.0,
            ..Default::default()
        };
        let mut state = Box3dCharacterControllerState {
            velocity: Vec3::X * cfg.speed,
            ..Default::default()
        };

        apply_box3d_air_friction(&cfg, &mut state, 1.0 / 60.0);

        assert!(state.velocity.x < cfg.speed, "{:?}", state.velocity);
    }

    #[test]
    fn upward_speed_can_unground_character() {
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
        let mut state = Box3dCharacterControllerState {
            velocity: Vec3::Y * (cfg.unground_speed + 1.0),
            grounded: Some(Box3dCastHit {
                entity: None,
                distance: 0.0,
                point: Vec3::ZERO,
                normal: Vec3::Y,
                collision_distance: 0.0,
            }),
            ..Default::default()
        };

        update_box3d_grounded(
            &runtime,
            &cfg,
            &mut state,
            &Transform::from_xyz(0.0, cfg.height * 0.5, 0.0),
            1.0 / 60.0,
        );

        assert!(state.grounded.is_none());
    }

    #[test]
    fn climb_factor_uses_input_and_look_pitch() {
        let cfg = Box3dCharacterController::default();
        let mut input = AccumulatedInput {
            last_movement: Some(bevy_math::Vec2::Y),
            ..Default::default()
        };
        let look = CharacterLook {
            pitch: 0.0,
            ..Default::default()
        };

        let forward = calculate_box3d_climb_factor(&cfg, &input, &look, Vec3::NEG_Z * cfg.speed);
        assert!(forward > 0.0, "{forward}");

        input.last_movement = Some(bevy_math::Vec2::NEG_Y);
        let backward = calculate_box3d_climb_factor(&cfg, &input, &look, Vec3::Z * cfg.speed);
        assert!(backward < 0.0, "{backward}");

        input.last_movement = Some(bevy_math::Vec2::Y);
        let looking_down = CharacterLook {
            pitch: -std::f32::consts::FRAC_PI_2,
            ..Default::default()
        };
        let stalled =
            calculate_box3d_climb_factor(&cfg, &input, &looking_down, Vec3::NEG_Z * cfg.speed);
        assert!(stalled < forward, "{stalled} >= {forward}");
    }

    #[test]
    fn mantle_probe_finds_clear_ledge() {
        let mut runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let ledge = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, 0.75, -1.0)));
        let ledge_shape = ledge.create_box(
            box3d::Vec3::new(2.0, 0.75, 0.25),
            box3d::ShapeDef::default(),
        );
        let ledge_entity = Entity::from_bits(42);
        runtime
            .shape_entities
            .insert(ledge_shape.id().to_bits(), ledge_entity);

        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState::default();
        let mut input = AccumulatedInput {
            mantled: Some(Stopwatch::new()),
            ..Default::default()
        };
        update_box3d_mantle_state(
            &runtime,
            &cfg,
            &mut state,
            &mut input,
            &Transform::from_xyz(0.0, cfg.height * 0.5, 0.0),
            Vec3::NEG_Z * cfg.speed,
        );

        let mantle = state.mantle.unwrap();
        assert_eq!(mantle.wall_entity, ledge_entity);
        assert!(mantle.target_position.y > cfg.height * 0.5);
        assert!(input.mantled.is_none());

        let mut state = Box3dCharacterControllerState::default();
        let mut input = AccumulatedInput {
            craned: Some(Stopwatch::new()),
            mantled: Some(Stopwatch::new()),
            jumped: Some(Stopwatch::new()),
            ..Default::default()
        };
        update_box3d_crane_state(
            &runtime,
            &cfg,
            &mut state,
            &mut input,
            &Transform::from_xyz(0.0, cfg.height * 0.5, 0.0),
            Vec3::NEG_Z * cfg.speed,
        );

        assert!(state.mantle.is_none());
        assert!(state.crane_height_left.is_some());
        assert!(input.craned.is_none());
        assert!(input.mantled.is_none());
        assert!(input.jumped.is_none());
    }

    #[test]
    fn climbdown_enters_mantle_from_empty_ledge() {
        let mut runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let ledge = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, 0.75, -1.0)));
        let ledge_shape = ledge.create_box(
            box3d::Vec3::new(2.0, 0.75, 0.25),
            box3d::ShapeDef::default(),
        );
        let ledge_entity = Entity::from_bits(42);
        runtime
            .shape_entities
            .insert(ledge_shape.id().to_bits(), ledge_entity);

        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState::default();
        let mut input = AccumulatedInput {
            last_movement: Some(bevy_math::Vec2::NEG_Y),
            climbdown: Some(Stopwatch::new()),
            ..Default::default()
        };
        let mut transform = Transform::from_xyz(0.0, cfg.height * 0.5 + cfg.crane_height, 0.0);
        let original_position = transform.translation;

        update_box3d_climbdown_state(
            &runtime,
            &cfg,
            &mut state,
            &mut input,
            &mut transform,
            Vec3::Z * cfg.speed,
        );

        let mantle = state.mantle.unwrap();
        assert_eq!(mantle.wall_entity, ledge_entity);
        assert_eq!(transform.translation, original_position);
        assert!(input.climbdown.is_none());
        assert!(input.jumped.is_none());
    }

    #[test]
    fn ledge_jump_releases_mantle() {
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            mantle: Some(Box3dMantleState {
                wall_normal: Dir3::Z,
                ledge_position: Vec3::ZERO,
                wall_entity: Entity::from_bits(42),
                target_position: Vec3::Y,
                speed: cfg.mantle_speed,
                automatic: false,
            }),
            ..Default::default()
        };
        let mut mantle_input = Stopwatch::new();
        mantle_input.set_elapsed(cfg.mantle_input_buffer);
        let mut input = AccumulatedInput {
            last_movement: Some(bevy_math::Vec2::Y),
            jumped: Some(Stopwatch::new()),
            mantled: Some(mantle_input),
            ..Default::default()
        };

        handle_box3d_ledge_jump(&cfg, &mut state, &mut input, &CharacterLook::default());

        assert!(state.mantle.is_none());
        assert!(state.velocity.y > 0.0);
        assert!(state.velocity.z < 0.0);
        assert!(input.jumped.is_none());
        assert!(input.mantled.is_none());
    }

    #[test]
    fn held_jump_does_not_release_fresh_mantle() {
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            mantle: Some(Box3dMantleState {
                wall_normal: Dir3::Z,
                ledge_position: Vec3::ZERO,
                wall_entity: Entity::from_bits(42),
                target_position: Vec3::Y,
                speed: cfg.mantle_speed,
                automatic: false,
            }),
            ..Default::default()
        };
        let mut input = AccumulatedInput {
            last_movement: Some(bevy_math::Vec2::Y),
            jumped: Some(Stopwatch::new()),
            mantled: None,
            ..Default::default()
        };

        handle_box3d_ledge_jump(&cfg, &mut state, &mut input, &CharacterLook::default());

        assert!(state.mantle.is_some());
        assert_eq!(state.velocity, Vec3::ZERO);
        assert!(input.jumped.is_some());
    }

    #[test]
    fn movement_crosses_adjacent_floor_colliders() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let left = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(-2.5, -0.5, 0.0)));
        let _left_shape =
            left.create_box(box3d::Vec3::new(2.5, 0.5, 5.0), box3d::ShapeDef::default());
        let right = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(
                2.5, -0.4999, 0.0,
            )));
        let _right_shape =
            right.create_box(box3d::Vec3::new(2.5, 0.5, 5.0), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            velocity: Vec3::X,
            ..Default::default()
        };
        let mut output = CharacterControllerOutput::default();
        let mut transform = Transform::from_xyz(-0.5, cfg.height * 0.5, 0.0);

        box3d_move_and_slide(
            &runtime,
            &cfg,
            &mut state,
            &mut output,
            &mut transform,
            Vec3::X,
        );

        assert!(transform.translation.x > 0.45, "{transform:?}");
        assert!(state.velocity.x > 0.99, "{:?}", state.velocity);
    }

    #[test]
    fn movement_slides_across_adjacent_wall_colliders() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let front = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(1.25, 1.0, 2.5)));
        let _front_shape =
            front.create_box(box3d::Vec3::new(0.5, 2.0, 2.5), box3d::ShapeDef::default());
        let back = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(1.25, 1.0, -2.5)));
        let _back_shape =
            back.create_box(box3d::Vec3::new(0.5, 2.0, 2.5), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            velocity: Vec3::NEG_Z,
            ..Default::default()
        };
        let mut output = CharacterControllerOutput::default();
        let mut transform = Transform::from_xyz(0.05, cfg.height * 0.5, 0.5);

        box3d_move_and_slide(
            &runtime,
            &cfg,
            &mut state,
            &mut output,
            &mut transform,
            Vec3::NEG_Z,
        );

        assert!(transform.translation.z < -0.45, "{transform:?}");
        assert!(state.velocity.z < -0.99, "{:?}", state.velocity);
    }

    #[test]
    fn sustained_corner_push_does_not_jitter_position() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let left_wall = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(-1.25, 1.0, 0.0)));
        let _left_shape =
            left_wall.create_box(box3d::Vec3::new(0.5, 2.0, 2.0), box3d::ShapeDef::default());
        let front_wall = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, 1.0, -1.25)));
        let _front_shape =
            front_wall.create_box(box3d::Vec3::new(2.0, 2.0, 0.5), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            velocity: Vec3::new(-cfg.speed, 0.0, -cfg.speed).normalize() * cfg.speed,
            ..Default::default()
        };
        let mut output = CharacterControllerOutput::default();
        let mut transform = Transform::from_xyz(0.0, cfg.height * 0.5, 0.0);
        for tick in 0..8 {
            let previous = transform.translation;
            box3d_move_and_slide(
                &runtime,
                &cfg,
                &mut state,
                &mut output,
                &mut transform,
                Vec3::new(-1.0, 0.0, -1.0).normalize() * 0.2,
            );
            state.velocity = Vec3::new(-cfg.speed, 0.0, -cfg.speed).normalize() * cfg.speed;

            if tick > 5 {
                assert!(
                    (transform.translation - previous).length() <= 1.0e-4,
                    "corner push moved too far: previous={previous:?}, current={:?}",
                    transform.translation
                );
            }
        }
    }

    #[test]
    fn zero_distance_corner_hit_does_not_apply_skin_shove() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let left_wall = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(-1.25, 1.0, 0.0)));
        let _left_shape =
            left_wall.create_box(box3d::Vec3::new(0.5, 2.0, 2.0), box3d::ShapeDef::default());
        let front_wall = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, 1.0, -1.25)));
        let _front_shape =
            front_wall.create_box(box3d::Vec3::new(2.0, 2.0, 0.5), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            velocity: Vec3::new(-cfg.speed, 0.0, -cfg.speed).normalize() * cfg.speed,
            ..Default::default()
        };
        let mut output = CharacterControllerOutput::default();
        let mut transform = Transform::from_xyz(
            -0.05 - cfg.skin_width,
            cfg.height * 0.5,
            -0.05 - cfg.skin_width,
        );
        let original = transform.translation;

        box3d_move_and_slide(
            &runtime,
            &cfg,
            &mut state,
            &mut output,
            &mut transform,
            Vec3::new(-1.0, 0.0, -1.0).normalize() * 0.2,
        );

        assert!(
            (transform.translation - original).length() <= 1.0e-4,
            "zero-distance corner hit moved: original={original:?}, current={:?}",
            transform.translation
        );
        assert!(state.velocity.x.abs() <= 1.0e-4, "{:?}", state.velocity);
        assert!(state.velocity.z.abs() <= 1.0e-4, "{:?}", state.velocity);
    }

    #[test]
    fn grounded_corner_push_does_not_jitter_position() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let floor = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, -0.5, 0.0)));
        let _floor_shape =
            floor.create_box(box3d::Vec3::new(5.0, 0.5, 5.0), box3d::ShapeDef::default());
        let left_wall = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(-1.25, 1.0, 0.0)));
        let _left_shape =
            left_wall.create_box(box3d::Vec3::new(0.5, 2.0, 2.0), box3d::ShapeDef::default());
        let front_wall = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, 1.0, -1.25)));
        let _front_shape =
            front_wall.create_box(box3d::Vec3::new(2.0, 2.0, 0.5), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let delta = 1.0 / 60.0;
        let wish_velocity = Vec3::new(-1.0, 0.0, -1.0).normalize() * cfg.speed;
        let mut state = Box3dCharacterControllerState::default();
        let mut output = CharacterControllerOutput::default();
        let mut transform = Transform::from_xyz(0.0, cfg.height * 0.5, 0.0);
        update_box3d_grounded(&runtime, &cfg, &mut state, &transform, delta);

        for tick in 0..16 {
            let previous = transform.translation;
            apply_box3d_friction(&runtime, &cfg, &mut state, delta);
            ground_accelerate(&cfg, &mut state, wish_velocity, delta);
            box3d_ground_move(
                &runtime,
                &cfg,
                &mut state,
                &mut output,
                &mut transform,
                delta,
            );
            update_box3d_grounded(&runtime, &cfg, &mut state, &transform, delta);

            if tick > 2 {
                assert!(
                    (transform.translation - previous).length() <= 1.0e-4,
                    "corner push moved too far: previous={previous:?}, current={:?}, velocity={:?}",
                    transform.translation,
                    state.velocity
                );
            }
        }
    }

    #[test]
    fn ceiling_hit_does_not_push_character_through_floor() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let floor = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, -0.5, 0.0)));
        let _floor_shape =
            floor.create_box(box3d::Vec3::new(5.0, 0.5, 5.0), box3d::ShapeDef::default());
        let ceiling = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, 1.95, 0.0)));
        let _ceiling_shape =
            ceiling.create_box(box3d::Vec3::new(5.0, 0.05, 5.0), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            velocity: Vec3::Y * 4.0,
            ..Default::default()
        };
        let mut output = CharacterControllerOutput::default();
        let mut transform = Transform::from_xyz(0.0, cfg.height * 0.5, 0.0);

        box3d_move_and_slide(
            &runtime,
            &cfg,
            &mut state,
            &mut output,
            &mut transform,
            Vec3::Y * 0.3,
        );
        depenetrate_box3d_character(&runtime, &cfg, &state, &mut transform);

        let feet_y = transform.translation.y - cfg.height * 0.5;
        assert!(feet_y >= -0.0001, "{transform:?}");
        assert!(state.velocity.y <= 0.0001, "{:?}", state.velocity);
    }

    #[test]
    fn ceiling_contact_clips_upward_velocity() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let floor = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, -0.5, 0.0)));
        let _floor_shape =
            floor.create_box(box3d::Vec3::new(5.0, 0.5, 5.0), box3d::ShapeDef::default());
        let ceiling = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, 1.85, 0.0)));
        let _ceiling_shape =
            ceiling.create_box(box3d::Vec3::new(5.0, 0.05, 5.0), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            velocity: Vec3::Y * 4.0,
            ..Default::default()
        };
        let mut output = CharacterControllerOutput::default();
        let mut transform = Transform::from_xyz(0.0, cfg.height * 0.5, 0.0);

        box3d_move_and_slide(
            &runtime,
            &cfg,
            &mut state,
            &mut output,
            &mut transform,
            Vec3::Y * 0.2,
        );

        let feet_y = transform.translation.y - cfg.height * 0.5;
        assert!(feet_y >= -0.0001, "{transform:?}");
        assert!(state.velocity.y <= 0.0001, "{:?}", state.velocity);
    }

    #[test]
    fn downward_overhang_plane_does_not_add_downward_velocity() {
        let planes = [box3d::CollisionPlane::rigid(box3d::Plane {
            normal: box3d::Vec3::new(-0.70710677, -0.70710677, 0.0),
            offset: 0.0,
        })];

        let clipped = clip_box3d_kcc_vector(Vec3::X, &planes);

        assert!(clipped.x >= -0.0001, "{clipped:?}");
        assert!(clipped.y >= -0.0001, "{clipped:?}");
    }

    #[test]
    fn clipping_oblique_corner_does_not_reenter_previous_plane() {
        let planes = [
            box3d::CollisionPlane::rigid(box3d::Plane {
                normal: box3d::Vec3::new(1.0, 0.0, 0.0),
                offset: 0.0,
            }),
            box3d::CollisionPlane::rigid(box3d::Plane {
                normal: box3d::Vec3::new(-0.70710677, 0.0, 0.70710677),
                offset: 0.0,
            }),
        ];
        let clipped = clip_box3d_kcc_vector(Vec3::new(-1.0, 0.0, -1.0), &planes);

        for plane in planes {
            let normal = from_box3d_vec3(plane.plane().normal).normalize_or_zero();
            assert!(clipped.dot(normal) >= -1.0e-4, "{clipped:?}");
        }
    }

    #[test]
    fn head_height_platform_contact_does_not_inject_downward_motion() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let floor = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, -0.5, 0.0)));
        let _floor_shape =
            floor.create_box(box3d::Vec3::new(5.0, 0.5, 5.0), box3d::ShapeDef::default());
        let platform = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(1.35, 0.82, 0.0)));
        let _platform_shape = platform.create_box(
            box3d::Vec3::new(0.25, 0.82, 2.0),
            box3d::ShapeDef::default(),
        );
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            velocity: Vec3::X * 4.0,
            ..Default::default()
        };
        let mut output = CharacterControllerOutput::default();
        let mut transform = Transform::from_xyz(0.0, cfg.height * 0.5, 0.0);

        update_box3d_grounded(&runtime, &cfg, &mut state, &transform, 1.0 / 60.0);
        box3d_move_and_slide(
            &runtime,
            &cfg,
            &mut state,
            &mut output,
            &mut transform,
            Vec3::X * 0.8,
        );
        update_box3d_grounded(&runtime, &cfg, &mut state, &transform, 1.0 / 60.0);

        let feet_y = transform.translation.y - cfg.height * 0.5;
        assert!(feet_y >= -0.0001, "{transform:?}");
        assert!(state.velocity.y >= -0.0001, "{:?}", state.velocity);
        assert!(state.grounded.is_some(), "{state:?}");
    }

    #[test]
    fn low_overhead_platform_keeps_character_on_moving_platform() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let moving = runtime
            .world
            .create_body(box3d::BodyDef::kinematic_at(box3d::Vec3::new(
                0.0, 1.5, -7.0,
            )));
        let _moving_shape =
            moving.create_box(box3d::Vec3::new(1.5, 0.15, 1.5), box3d::ShapeDef::default());
        let overhead = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(
                -2.0, 1.75, -7.0,
            )));
        let _overhead_shape =
            overhead.create_box(box3d::Vec3::new(2.0, 0.15, 1.5), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let platform_top = 1.65;
        let mut state = Box3dCharacterControllerState {
            velocity: Vec3::NEG_X * 12.0,
            grounded: Some(Box3dCastHit {
                entity: None,
                distance: 0.0,
                point: Vec3::new(0.0, platform_top, -7.0),
                normal: Vec3::Y,
                collision_distance: 0.0,
            }),
            ..Default::default()
        };
        let mut output = CharacterControllerOutput::default();
        let mut transform = Transform::from_xyz(0.8, platform_top + cfg.height * 0.5, -7.0);

        box3d_ground_move(
            &runtime,
            &cfg,
            &mut state,
            &mut output,
            &mut transform,
            1.0 / 60.0,
        );
        update_box3d_grounded(&runtime, &cfg, &mut state, &transform, 1.0 / 60.0);

        let feet_y = transform.translation.y - cfg.height * 0.5;
        assert!(feet_y >= platform_top - 0.0001, "{transform:?}");
        assert!(state.grounded.is_some(), "{state:?}");
    }

    #[test]
    fn grounded_depenetration_does_not_push_character_below_moving_platform() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let moving = runtime
            .world
            .create_body(box3d::BodyDef::kinematic_at(box3d::Vec3::new(
                0.0, 1.5, -7.0,
            )));
        let _moving_shape =
            moving.create_box(box3d::Vec3::new(1.5, 0.15, 1.5), box3d::ShapeDef::default());
        let overhead = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(
                -2.0, 1.75, -7.0,
            )));
        let _overhead_shape =
            overhead.create_box(box3d::Vec3::new(2.0, 0.15, 1.5), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let platform_top = 1.65;
        let mut state = Box3dCharacterControllerState {
            grounded: Some(Box3dCastHit {
                entity: None,
                distance: 0.0,
                point: Vec3::new(0.0, platform_top, -7.0),
                normal: Vec3::Y,
                collision_distance: 0.0,
            }),
            ..Default::default()
        };
        let mut transform = Transform::from_xyz(0.65, platform_top + cfg.height * 0.5, -7.0);

        depenetrate_box3d_character(&runtime, &cfg, &state, &mut transform);
        update_box3d_grounded(&runtime, &cfg, &mut state, &transform, 1.0 / 60.0);

        let feet_y = transform.translation.y - cfg.height * 0.5;
        assert!(feet_y >= platform_top - 0.0001, "{transform:?}");
        assert!(state.grounded.is_some(), "{state:?}");
    }

    #[test]
    fn grounded_probe_accepts_tiny_floor_overlap() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let moving = runtime
            .world
            .create_body(box3d::BodyDef::kinematic_at(box3d::Vec3::new(
                0.0, 1.5, -7.0,
            )));
        let _moving_shape =
            moving.create_box(box3d::Vec3::new(1.5, 0.15, 1.5), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let platform_top = 1.65;
        let mut state = Box3dCharacterControllerState::default();
        let transform = Transform::from_xyz(0.0, platform_top + cfg.height * 0.5 - 0.002, -7.0);

        update_box3d_grounded(&runtime, &cfg, &mut state, &transform, 1.0 / 60.0);

        assert!(state.grounded.is_some(), "{state:?}");
        let ground = state.grounded.unwrap();
        assert!(ground.normal.y >= cfg.min_walk_cos, "{ground:?}");
    }

    #[test]
    fn low_head_platform_contact_keeps_floor_grounding() {
        let runtime = Box3dRuntime::new(AhoyBox3dConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let floor = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(0.0, -0.5, 0.0)));
        let _floor_shape = floor.create_box(
            box3d::Vec3::new(10.0, 0.5, 10.0),
            box3d::ShapeDef::default(),
        );
        let overhead = runtime
            .world
            .create_body(box3d::BodyDef::static_at(box3d::Vec3::new(
                -2.0, 1.75, -7.0,
            )));
        let _overhead_shape =
            overhead.create_box(box3d::Vec3::new(2.0, 0.15, 1.5), box3d::ShapeDef::default());
        let cfg = Box3dCharacterController::default();
        let mut state = Box3dCharacterControllerState {
            velocity: Vec3::NEG_X * 12.0,
            ..Default::default()
        };
        let mut output = CharacterControllerOutput::default();
        let mut transform = Transform::from_xyz(0.8, cfg.height * 0.5, -7.0);

        update_box3d_grounded(&runtime, &cfg, &mut state, &transform, 1.0 / 60.0);
        box3d_ground_move(
            &runtime,
            &cfg,
            &mut state,
            &mut output,
            &mut transform,
            1.0 / 60.0,
        );
        update_box3d_grounded(&runtime, &cfg, &mut state, &transform, 1.0 / 60.0);

        let feet_y = transform.translation.y - cfg.height * 0.5;
        assert!(feet_y >= -0.0001, "{transform:?}");
        assert!(state.grounded.is_some(), "{state:?}");
    }

    #[test]
    fn push_impulse_uses_reduced_effective_mass() {
        let world = box3d::World::new(box3d::Vec3::ZERO);
        let body = world.create_body(box3d::BodyDef::dynamic_at(box3d::Vec3::ZERO));
        let _shape = body.create_box(
            box3d::Vec3::new(0.4, 0.4, 0.4),
            box3d::ShapeDef {
                density: 1.0,
                ..Default::default()
            },
        );
        let body_id = body.id();

        let impulse =
            calculate_box3d_push_impulse(body_id, 80.0, Vec3::ZERO, Vec3::X, Vec3::X * 12.0);
        assert!(impulse.x > 0.0 && impulse.x < 10.0, "{impulse:?}");

        body_id.apply_linear_impulse(to_box3d_vec3(impulse), box3d::Vec3::ZERO, true);
        let velocity = from_box3d_vec3(body_id.linear_velocity());
        assert!(velocity.x > 0.0 && velocity.x <= 12.0, "{velocity:?}");

        let separating =
            calculate_box3d_push_impulse(body_id, 80.0, Vec3::ZERO, Vec3::X, Vec3::NEG_X);
        assert_eq!(separating, Vec3::ZERO);
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
