//!
//! # Random Forest learning
//! `linfa-forest` aims to provide pure rust implementations
//! of random forest learning algorithms.
//!
//! # The big picture
//!
//! `linfa-forest` is a crate in the [linfa](https://github.com/rust-ml/linfa) ecosystem,
//! an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's scikit-learn.
//!Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks
//! that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of
//! the random forest is the class selected by most trees. For regression tasks, the mean or average prediction
//! of the individual trees is returned.Random decision forests correct for decision trees' habit of overfitting
//! to their training set
//!
//! # Current state
//!
//! `linfa-forest` currently provides an [implementation](struct.RandomForest.html) of tree fitting for classification.
//!

mod random_forest;

pub use random_forest::*;
pub use linfa::error::Result;
