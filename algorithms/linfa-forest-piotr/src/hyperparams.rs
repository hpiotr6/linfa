// use linfa::{
//     error::{Error, Result},
//     Float, Label, ParamGuard,
// };
use std::marker::PhantomData;
use linfa::{
    error::{Error, Result},
    Float, Label, ParamGuard,
};
use linfa_trees::SplitQuality;

use crate::algorithm::RandomForest;

pub struct RandomForestValidParams<F, L> {
    split_quality: SplitQuality,
    max_depth: Option<usize>,
    min_weight_split: f32,
    min_weight_leaf: f32,
    min_impurity_decrease: F,

    label_marker: PhantomData<L>,
}

impl<F: Float, L> RandomForestValidParams<F, L> {
    pub fn split_quality(&self) -> SplitQuality {
        self.split_quality
    }

    pub fn max_depth(&self) -> Option<usize> {
        self.max_depth
    }

    pub fn min_weight_split(&self) -> f32 {
        self.min_weight_split
    }

    pub fn min_weight_leaf(&self) -> f32 {
        self.min_weight_leaf
    }

    pub fn min_impurity_decrease(&self) -> F {
        self.min_impurity_decrease
    }
}

pub struct RandomForestParams<F, L>(RandomForestValidParams<F, L>);

impl<F:Float, L:Label> RandomForestParams<F,L>{
    pub fn new() -> Self {
        Self(RandomForestValidParams {
            split_quality: SplitQuality::Gini,
            max_depth: None,
            min_weight_split: 2.0,
            min_weight_leaf: 1.0,
            min_impurity_decrease: F::cast(0.00001),
            label_marker: PhantomData,
        })
    }
    /// Sets the metric used to decide the feature on which to split a node
    pub fn split_quality(mut self, split_quality: SplitQuality) -> Self {
        self.0.split_quality = split_quality;
        self
    }

    /// Sets the optional limit to the depth of the decision tree
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.0.max_depth = max_depth;
        self
    }

    /// Sets the minimum weight of samples required to split a node.
    ///
    /// If the observations do not have associated weights, this value represents
    /// the minimum number of samples required to split a node.
    pub fn min_weight_split(mut self, min_weight_split: f32) -> Self {
        self.0.min_weight_split = min_weight_split;
        self
    }

    /// Sets the minimum weight of samples that a split has to place in each leaf
    ///
    /// If the observations do not have associated weights, this value represents
    /// the minimum number of samples that a split has to place in each leaf.
    pub fn min_weight_leaf(mut self, min_weight_leaf: f32) -> Self {
        self.0.min_weight_leaf = min_weight_leaf;
        self
    }

    /// Sets the minimum decrease in impurity that a split needs to bring in order for it to be applied
    pub fn min_impurity_decrease(mut self, min_impurity_decrease: F) -> Self {
        self.0.min_impurity_decrease = min_impurity_decrease;
        self
    }

}


impl<F: Float, L: Label> Default for RandomForestParams<F, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float, L: Label> RandomForest<F, L> {
    /// Defaults are provided if the optional parameters are not specified:
    /// * `split_quality = SplitQuality::Gini`
    /// * `max_depth = None`
    /// * `min_weight_split = 2.0`
    /// * `min_weight_leaf = 1.0`
    /// * `min_impurity_decrease = 0.00001`
    // Violates the convention that new should return a value of type `Self`
    #[allow(clippy::new_ret_no_self)]
    pub fn params() -> RandomForestParams<F, L> {
        RandomForestParams::new()
    }
}

impl<F: Float, L> ParamGuard for RandomForestParams<F, L> {
    type Checked = RandomForestValidParams<F, L>;
    type Error = Error;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.min_impurity_decrease < F::epsilon() {
            Err(Error::Parameters(format!(
                "Minimum impurity decrease should be greater than zero, but was {}",
                self.0.min_impurity_decrease
            )))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}


