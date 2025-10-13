// Definitions that are used throughout all modules

// Enumeration for dimensionality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    _2D,
    _3D,
}

// Enumeration to track the space type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Space {
    Real,
    Reciprocal,
}

pub mod space {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Direct;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Reciprocal;
}
