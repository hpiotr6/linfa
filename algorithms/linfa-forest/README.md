# Random Forest learning
`linfa-forest` aims to provide pure rust implementations
of random forest learning algorithms.

# The big picture

`linfa-forest` is a crate in the [linfa](https://github.com/rust-ml/linfa) ecosystem,
an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's scikit-learn.
Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks
that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of
the random forest is the class selected by most trees.

# Current state

`linfa-forest` currently provides an implementation of single-tree fitting for classification.


## Examples

You can find examples in the `examples/` directory. To run RandomForest example, use:

```bash
$ cargo run --example random_forest --release
```

<details>
<summary style="cursor: pointer; display:list-item;">
Show source code
</summary>

```rust, no_run
uuse ndarray_rand::rand::SeedableRng;
use rand::rngs::{StdRng};

use linfa::prelude::*;
use linfa_forest::{RandomForest, Result};
use linfa_trees::{DecisionTree, SplitQuality};

use linfa_forest::BootstrapType::BootstrapSamples;

let mut rng = StdRng::seed_from_u64(42);

// Load the dataset
let (train, test) = linfa_datasets::iris()
    .shuffle(&mut rng)
    .split_with_ratio(0.8);

// Create a sample tree to replicate in the forest
let tree_params = DecisionTree::params()
    .split_quality(SplitQuality::Gini)
    .max_depth(Some(10))
    .min_weight_split(1.0)
    .min_weight_leaf(1.0);

// Create and train the forest
let forest = RandomForest::params()
    .n_trees(5)
    .tree_params(tree_params)
    .bootstrap_type(BootstrapSamples(10))
    .bootstrap_rng(StdRng::seed_from_u64(6))
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



```
</details>


## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.
