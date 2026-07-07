//! Box3D interop.
//!
//! This module is enabled with the `box3d` feature and re-exports the
//! low-level [`box3d`] crate and the Bevy adapter [`bevy_box3d`].
//!
//! The current Ahoy controller implementation still runs on Avian because it
//! depends on Avian's move-and-slide queries and collider metadata. At the time
//! this feature was added, `bevy_box3d` targets Bevy 0.19 while Ahoy targets
//! Bevy 0.18, so the Box3D plugin types cannot be added to the same Bevy app
//! as Ahoy without upgrading the rest of the dependency stack.

pub use ::bevy_box3d;
pub use ::box3d;

/// Common Box3D imports for users experimenting with the `box3d` feature.
pub mod prelude {
    pub use super::{
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
