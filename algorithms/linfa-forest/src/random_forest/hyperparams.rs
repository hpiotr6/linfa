use linfa::{
    error::{Error, Result},
    Float, Label, ParamGuard,
};
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

/// The metric used to determine the feature by which a node is split
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug)]
pub enum SplitQuality {
    /// Measures the degree of probability of a randomly chosen point in the subtree being misclassified, defined as
    /// one minus the sum over all labels of the squared probability of encountering that label.
    /// The Gini index of the root is given by the weighted sum of the indexes of its two subtrees.
    /// At each step the split is applied to the feature which decreases the most the Gini impurity of the root.
    Gini,
    /// Measures the entropy of a subtree, defined as the sum over all labels of the probability of encountering that label in the
    /// subtree times its logarithm in base two, with negative sign. The entropy of the root minus the weighted sum of the entropy
    /// of its two subtrees defines the "information gain" obtained by applying the split. At each step the split is applied to the
    /// feature with the biggest information gain
    Entropy,
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug)]
pub struct RandomForestValidParams<F, L> {
    split_quality: SplitQuality,
    max_depth: Option<usize>,
    min_weight_split: f32,
    min_weight_leaf: f32,
    min_impurity_decrease: F,

    label_marker: PhantomData<L>,
}

impl<F: Float, L> RandomForestValidParams<F, L> {

}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug)]
pub struct RandomForestParams<F, L>(RandomForestValidParams<F, L>);

impl<F: Float, L: Label> RandomForestParams<F, L> {
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
}

impl<F: Float, L: Label> Default for RandomForestParams<F, L> {
    fn default() -> Self {
        Self::new()
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
