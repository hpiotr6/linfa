use std::fs::File;
use std::io::Write;

use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;

use linfa::prelude::*;
use linfa_forest::{RandomForest, Result};
use linfa_trees::DecisionTree;

fn main() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(42);

    // Load the dataset
    let dataset = linfa_datasets::iris();
    // Fit the tree
    let tree = RandomForest::params().fit(&dataset).unwrap();
    // // Get accuracy on training set
    // let accuracy = tree.predict(&dataset).confusion_matrix(&dataset).unwrap().accuracy£();
    //


    Ok(())
}
