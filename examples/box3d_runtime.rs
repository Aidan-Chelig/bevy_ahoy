use bevy::prelude::*;
use bevy_ahoy::box3d::prelude::*;

fn main() -> AppExit {
    App::new()
        .add_plugins((
            DefaultPlugins,
            AhoyBox3dPlugin {
                config: AhoyBox3dConfig {
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    sub_steps: 8,
                },
            },
        ))
        .add_systems(Startup, setup)
        .run()
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-6.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let floor_mesh = meshes.add(Cuboid::new(12.0, 1.0, 12.0));
    let floor_material = materials.add(Color::srgb(0.25, 0.28, 0.32));
    commands.spawn((
        Mesh3d(floor_mesh),
        MeshMaterial3d(floor_material),
        Transform::from_xyz(0.0, -0.5, 0.0),
        AhoyBox3dBody::STATIC,
        AhoyBox3dCollider::cuboid(Vec3::new(6.0, 0.5, 6.0)),
    ));

    let ball_mesh = meshes.add(Sphere::new(0.5));
    let ball_material = materials.add(Color::srgb(0.1, 0.65, 0.95));
    for i in 0..8 {
        commands.spawn((
            Mesh3d(ball_mesh.clone()),
            MeshMaterial3d(ball_material.clone()),
            Transform::from_xyz(-3.5 + i as f32, 2.0 + i as f32 * 0.7, 0.0),
            AhoyBox3dBody::DYNAMIC,
            AhoyBox3dCollider::sphere(0.5),
            AhoyBox3dVelocity::default(),
        ));
    }
}
