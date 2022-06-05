use ndarray_rand::rand::SeedableRng;
use rand::rngs::{SmallRng, StdRng};

use linfa::prelude::*;
use linfa_forest::{RandomForest, Result};
use linfa_trees::{DecisionTree, SplitQuality};

use linfa_forest::BootstrapType::BootstrapSamples;

fn main() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(42);

    // Load the dataset
    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    // Create a sample tree to replicate in the forest
    let tree_params = DecisionTree::params()
        .split_quality(SplitQuality::Entropy)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0);

    // Create and train the forest
    let forest = RandomForest::params()
        .n_trees(5)
        .tree_params(tree_params)
        .bootstrap_type(BootstrapSamples(10))
        .rng(StdRng::from_rng(SmallRng::seed_from_u64(20)).unwrap())
        .fit(&train)?;

    // Get accuracy on training set
    let pred_train = forest.predict(&train);
    let cm_train = pred_train.confusion_matrix(&train)?;
    println!("{:?}", cm_train);
    println!("Training set accuracy: {:.2}%", 100.0 * cm_train.accuracy());

    // Get accuracy on test set
    let pred_test = forest.predict(&test);
    let cm_test = pred_test.confusion_matrix(&test)?;
    println!("{:?}", cm_test);
    println!("Test set accuracy: {:.2}%", 100.0 * cm_test.accuracy());



    Ok(())

}
