//! Linear decision trees
//!
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use linfa::dataset::AsSingleTargets;
use ndarray::{Array1, ArrayBase, Axis, Data, Ix1, Ix2};

use super::{RandomForestValidParams, SplitQuality};
use linfa::{
    dataset::{Labels, Records},
    error::Error,
    error::Result,
    traits::*,
    DatasetBase, Float, Label,
};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};



