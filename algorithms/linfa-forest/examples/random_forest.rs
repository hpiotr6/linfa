use std::fs::File;
use std::io::Write;

use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;

use linfa::prelude::*;
use linfa_forest::{RandomForest, Result};
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::{ArrayBase, OwnedRepr, Array2, Ix2, Array, Dimension};

fn main() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(42);

    // Load the dataset
    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    // Fit forest
    let tree_params = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0);

    let forest = RandomForest::params()
        .n_trees(100)
        .tree_params(tree_params)
        .fit(&train)?;

    // // Get accuracy on training set
    let cm = forest.predict(&test).confusion_matrix(&test)?;
    println!("{:?}", cm);
    println!("Test accuracy: {:.2}%",100.0 * cm.accuracy());
    //
    // let pred = forest.predict(&test);

    // // ocena
    // let mut acc: f64 = 0.0;
    // for (i, prediction) in pred.iter().enumerate() {
    //     // println!("i:{i}, predicted:[{}], real:[{}]", prediction, test.targets.row(i).index(0));
    //     acc += if prediction == test.targets.row(i).index(0) {1.0} else {0.0};
    // }
    // println!("ACC:[{}%]", acc = 100.0 * (acc / pred.len() as f64));

    Ok(())

}
