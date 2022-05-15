//! Linear decision trees
//!
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use linfa::dataset::{AsSingleTargets, AsTargets, FromTargetArray};
use ndarray::{Array1, ArrayBase, Axis, Data, DataOwned, Dimension, Ix1, Ix2};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use super::{RandomForestValidParams, BootstrapType};
use linfa::{
    dataset::{Labels, Records},
    error::Error,
    error::Result,
    traits::*,
    DatasetBase, Float, Label,
};
use linfa_trees::{DecisionTree, DecisionTreeParams, DecisionTreeValidParams};


#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};


pub struct RandomForest<F: Float, L: Label> {
    n_trees: usize,
    tree_params: DecisionTreeParams<F, L>,
    bootstrap_type: BootstrapType,
    trees: Vec<DecisionTree<F, L>>
}


impl<'a, F: Float, L: Label + 'a + std::fmt::Debug, D, T> Fit<ArrayBase<D, Ix2>, T, Error>
for RandomForestValidParams<F, L>
    where
        D: Data<Elem = F>,
        T: AsSingleTargets<Elem = L> + Labels<Elem = L>,



// FIXME: funkcje dataset.bootstrap wymagaja tych traitow (moze da sie to zapisac lepiej?)
// where T: FromTargetArray<'a> ,
//       <T as FromTargetArray<'a>>::Owned: AsTargets,
//       <T as AsTargets>::Elem: Copy
{
    type Object = RandomForest<F, L>;


    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let mut rng = SmallRng::seed_from_u64(42); // TODO: rng jako parametr lasu
        // let x = dataset.bootstrap_samples(10, &mut rng);

        // FIXME: wybór rodzaju bootstrapa nie dziala, kompilator nie pozwala
        // let mut subsample_iterator = match self.bootstrap_type() {
        //     BootstrapType::BootstrapSamples(n_samples) => dataset.bootstrap_samples(n_samples, &mut rng),
        //     BootstrapType::BootstrapFeatures(n_features) => dataset.bootstrap_features(n_features, &mut rng),
        //     BootstrapType::BootstrapSamplesFeatures(n_samples, n_features) => dataset.bootstrap((n_samples, n_features), &mut rng)
        // };
        let mut trees: Vec<DecisionTree<F, L>> = Vec::new();

        for _i in 0..self.n_trees() {
            // let subsample = subsample_iterator.next().ok_or(Err("Failed to create subsample iterator"));
            trees.push(self.tree_params().fit(&dataset)?);
        }

    Ok(RandomForest{
        n_trees: self.n_trees(),
        tree_params: self.tree_params().clone(),
        bootstrap_type: self.bootstrap_type(),
        trees: trees
    })
     }
}

impl<F: Float, L: Label + Default, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
    for RandomForest<F, L> {

    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        todo!()
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    //
    use approx::assert_abs_diff_eq;
    use linfa::{error::Result, metrics::ToConfusionMatrix, Dataset, ParamGuard};
    use ndarray::{array, concatenate, s, Array, Array1, Array2, Axis};
    use rand::rngs::SmallRng;
    //
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};

    #[test]
    /// Check that for random data the n trees is used
    fn check_n_trees() -> Result<()> {
        let mut rng = SmallRng::seed_from_u64(42);

        // create very sparse data
        let data = Array::random_using((50, 50), Uniform::new(-1., 1.), &mut rng);
        let targets = (0..50).collect::<Array1<usize>>();
        //
        let dataset = Dataset::new(data, targets);
        //
        // // check that the provided depth is actually used
        // for max_depth in &[1, 5, 10, 20] {
        //     let model = DecisionTree::params()
        //         .max_depth(Some(*max_depth))
        //         .min_impurity_decrease(1e-10f64)
        //         .min_weight_split(1e-10)
        //         .fit(&dataset)?;
        //     assert_eq!(model.max_depth(), *max_depth);
        // }
        for trees_num in [1, 5, 20, 100]{
            let model = RandomForest::params().n_trees(trees_num).fit(&dataset)?;
            assert_eq!(model.n_trees, trees_num);

        }

        Ok(())
    }

    #[test]
    #[should_panic]
    /// Check that a zero trees param panics
    fn panic_min_impurity_decrease() {
        RandomForest::<f64, bool>::params()
            .n_trees(0)
            .check()
            .unwrap();
    }
}