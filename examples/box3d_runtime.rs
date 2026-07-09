use bevy::{
    input::common_conditions::input_just_pressed,
    prelude::*,
    window::{CursorGrabMode, CursorOptions},
};
use bevy_ahoy::{
    AhoyFixedUpdateUtilsPlugin, AhoyInputPlugin,
    box3d::prelude::*,
    camera::AhoyCameraPlugin,
    input::{Crouch, Jump, Movement, RotateCamera, SwimUp},
    prelude::CharacterControllerCameraOf,
};
use bevy_enhanced_input::prelude::*;

fn main() -> AppExit {
    App::new()
        .add_plugins((
            DefaultPlugins,
            EnhancedInputPlugin,
            AhoyFixedUpdateUtilsPlugin,
            AhoyInputPlugin,
            AhoyCameraPlugin,
            AhoyBox3dPlugin {
                config: AhoyBox3dConfig {
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    sub_steps: 8,
                },
            },
        ))
        .add_input_context::<PlayerInput>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                capture_cursor.run_if(input_just_pressed(MouseButton::Left)),
                release_cursor.run_if(input_just_pressed(KeyCode::Escape)),
            ),
        )
        .add_systems(FixedUpdate, reverse_platform)
        .run()
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    spawn_box(
        &mut commands,
        &mut meshes,
        &mut materials,
        Vec3::new(24.0, 1.0, 24.0),
        Transform::from_xyz(0.0, -0.5, 0.0),
        Color::srgb(0.25, 0.28, 0.32),
    );
    spawn_box(
        &mut commands,
        &mut meshes,
        &mut materials,
        Vec3::new(4.0, 0.4, 4.0),
        Transform::from_xyz(5.0, 0.2, 0.0)
            .with_rotation(Quat::from_rotation_z(-20.0_f32.to_radians())),
        Color::srgb(0.38, 0.43, 0.48),
    );
    for i in 0..6 {
        spawn_box(
            &mut commands,
            &mut meshes,
            &mut materials,
            Vec3::new(1.0, 0.18 + i as f32 * 0.08, 3.0),
            Transform::from_xyz(-5.0 + i as f32 * 1.05, 0.09 + i as f32 * 0.04, -4.0),
            Color::srgb(0.36, 0.4, 0.45),
        );
    }
    spawn_box(
        &mut commands,
        &mut meshes,
        &mut materials,
        Vec3::new(1.0, 2.0, 8.0),
        Transform::from_xyz(-8.0, 1.0, 0.0),
        Color::srgb(0.3, 0.35, 0.4),
    );
    spawn_box_with_friction(
        &mut commands,
        &mut meshes,
        &mut materials,
        Vec3::new(5.0, 0.08, 5.0),
        Transform::from_xyz(6.0, 0.04, -6.0),
        Color::srgb(0.45, 0.72, 0.82),
        0.05,
    );
    spawn_box(
        &mut commands,
        &mut meshes,
        &mut materials,
        Vec3::new(4.0, 0.3, 3.0),
        Transform::from_xyz(-2.0, 1.75, -7.0),
        Color::srgb(0.38, 0.4, 0.45),
    );
    for i in 0..4 {
        spawn_dynamic_box(
            &mut commands,
            &mut meshes,
            &mut materials,
            Vec3::splat(0.8),
            Transform::from_xyz(1.5 + i as f32 * 1.0, 0.4, 3.0),
            Color::srgb(0.65, 0.48, 0.28),
        );
    }
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(3.0, 0.3, 3.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.25, 0.55, 0.48))),
        Transform::from_xyz(0.0, 1.5, -7.0),
        AhoyBox3dBody::KINEMATIC,
        AhoyBox3dCollider::cuboid(Vec3::new(1.5, 0.15, 1.5)),
        AhoyBox3dVelocity::linear(Vec3::X * 1.5),
        MovingPlatform {
            left: -3.0,
            right: 3.0,
        },
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(5.0, 2.0, 5.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.08, 0.42, 0.68, 0.35),
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_xyz(7.0, 1.0, 5.0),
        Box3dWater::cuboid(Vec3::new(2.5, 1.0, 2.5)),
    ));

    let player = commands
        .spawn((
            Transform::from_xyz(0.0, 2.0, 6.0),
            Box3dCharacterController::default(),
            PlayerInput,
            actions!(PlayerInput[
                (
                    Action::<Movement>::new(),
                    DeadZone::default(),
                    Bindings::spawn((Cardinal::wasd_keys(), Axial::left_stick()))
                ),
                (
                    Action::<Jump>::new(),
                    bindings![KeyCode::Space, GamepadButton::South],
                ),
                (
                    Action::<SwimUp>::new(),
                    bindings![KeyCode::Space, GamepadButton::South],
                ),
                (
                    Action::<Crouch>::new(),
                    bindings![KeyCode::ControlLeft, GamepadButton::LeftTrigger2],
                ),
                (
                    Action::<RotateCamera>::new(),
                    Bindings::spawn((
                        Spawn((Binding::mouse_motion(), Scale::splat(0.07))),
                        Axial::right_stick().with((Scale::splat(4.0), DeadZone::default())),
                    ))
                ),
            ]),
        ))
        .id();

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.0, 8.0),
        CharacterControllerCameraOf::new(player),
    ));
}

#[derive(Component)]
struct MovingPlatform {
    left: f32,
    right: f32,
}

fn reverse_platform(mut platforms: Query<(&Transform, &MovingPlatform, &mut AhoyBox3dVelocity)>) {
    for (transform, platform, mut velocity) in &mut platforms {
        if (transform.translation.x >= platform.right && velocity.linear.x > 0.0)
            || (transform.translation.x <= platform.left && velocity.linear.x < 0.0)
        {
            velocity.linear.x = -velocity.linear.x;
        }
    }
}

fn spawn_box(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    size: Vec3,
    transform: Transform,
    color: Color,
) {
    spawn_box_with_friction(commands, meshes, materials, size, transform, color, 0.6);
}

fn spawn_box_with_friction(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    size: Vec3,
    transform: Transform,
    color: Color,
    friction: f32,
) {
    let mut collider = AhoyBox3dCollider::cuboid(size * 0.5);
    collider.friction = friction;
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::from_size(size))),
        MeshMaterial3d(materials.add(color)),
        transform,
        AhoyBox3dBody::STATIC,
        collider,
    ));
}

fn spawn_dynamic_box(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    size: Vec3,
    transform: Transform,
    color: Color,
) {
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::from_size(size))),
        MeshMaterial3d(materials.add(color)),
        transform,
        AhoyBox3dBody {
            linear_damping: 0.4,
            angular_damping: 0.4,
            ..AhoyBox3dBody::DYNAMIC
        },
        AhoyBox3dCollider::cuboid(size * 0.5),
    ));
}

#[derive(Component, Default)]
struct PlayerInput;

fn capture_cursor(mut cursor: Single<&mut CursorOptions>) {
    cursor.grab_mode = CursorGrabMode::Locked;
    cursor.visible = false;
}

fn release_cursor(mut cursor: Single<&mut CursorOptions>) {
    cursor.visible = true;
    cursor.grab_mode = CursorGrabMode::None;
}
