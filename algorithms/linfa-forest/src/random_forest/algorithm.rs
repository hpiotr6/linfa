//! Linear decision trees
//!
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

impl<'a, F: Float, L: Label, D: Data<Elem = F>, T>
Fit<ArrayBase<D, Ix2>, T, Error> for RandomForestValidParams<F, L>
// FIXME: funkcje dataset.bootstrap wymagaja tych traitow (moze da sie to zapisac lepiej?)
// where T: FromTargetArray<'a> ,
//       <T as FromTargetArray<'a>>::Owned: AsTargets,
//       <T as AsTargets>::Elem: Copy
{
    type Object = RandomForest<F, L>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let mut rng = SmallRng::seed_from_u64(42); // TODO: rng jako parametr lasu
        // let mut subsample_iterator = dataset.bootstrap_samples(10, &mut rng);

        // FIXME: wybór rodzaju bootstrapa nie dziala, kompilator nie pozwala
        // let mut subsample_iterator = match self.bootstrap_type() {
        //     BootstrapType::BootstrapSamples(n_samples) => dataset.bootstrap_samples(n_samples, &mut rng),
        //     BootstrapType::BootstrapFeatures(n_features) => dataset.bootstrap_features(n_features, &mut rng),
        //     BootstrapType::BootstrapSamplesFeatures(n_samples, n_features) => dataset.bootstrap((n_samples, n_features), &mut rng)
        // };
        for _i in 0..self.n_trees() {
            // let subsample = subsample_iterator.next().ok_or(Err("Failed to create subsample iterator"));
            // self.trees().push(self.tree_params().clone().fit(&subsample)?);
        }
        todo!()
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

