use bevy::{
    input::common_conditions::input_just_pressed,
    prelude::*,
    window::{CursorGrabMode, CursorOptions},
};
use bevy_ahoy::{
    AhoyFixedUpdateUtilsPlugin, AhoyInputPlugin,
    box3d::prelude::*,
    camera::AhoyCameraPlugin,
    input::{Jump, Movement, RotateCamera},
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

fn spawn_box(
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
        AhoyBox3dBody::STATIC,
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
