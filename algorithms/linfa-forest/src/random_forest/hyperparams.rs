use linfa::{
    error::{Error, Result},
    Float, Label, ParamGuard,
};
use linfa_trees::{DecisionTreeParams};


use rand::rngs::{StdRng};
use rand::SeedableRng;


#[derive(Clone, Copy, Debug)]
pub enum BootstrapType {
    BootstrapSamples(usize),
    BootstrapFeatures(usize),
    BootstrapSamplesFeatures(usize, usize)
}


#[derive(Clone, Debug)]
pub struct RandomForestValidParams<F: Float, L: Label> {
    n_trees: usize,
    tree_params: DecisionTreeParams<F, L>,
    bootstrap_type: BootstrapType,
    rng: StdRng
}

use crate::RandomForest;

impl<F: Float, L: Label> RandomForestValidParams<F, L> {

    pub fn n_trees(&self) -> usize {
        self.n_trees
    }
    pub fn bootstrap_type(&self) -> BootstrapType {
        self.bootstrap_type
    }
    pub fn tree_params(&self) -> &DecisionTreeParams<F, L> {
        &self.tree_params
    }
    pub fn rng(&self) -> &StdRng { &self.rng }

}

#[derive(Clone, Debug)]
pub struct RandomForestParams<F: Float, L: Label>(RandomForestValidParams<F, L>);

impl<F: Float, L: Label> RandomForestParams<F, L> {
    pub fn new() -> Self {
        Self(RandomForestValidParams {
            n_trees: 100,
            tree_params: DecisionTreeParams::new(),
            bootstrap_type: BootstrapType::BootstrapSamples(120),
            rng: StdRng::from_entropy()
        })
    }

    /// Sets number of trees in forest
    pub fn n_trees(mut self, n_trees: usize) -> Self {
        self.0.n_trees = n_trees;
        self
    }
    /// Sets parameters of decision trees (same to each other)
    pub fn tree_params(mut self, tree_params: DecisionTreeParams<F, L>) -> Self {
        self.0.tree_params = tree_params;
        self
    }
    /// Sets type of bootstrapping
    pub fn bootstrap_type(mut self, bootstrap_type: BootstrapType) -> Self {
        self.0.bootstrap_type = bootstrap_type;
        self
    }

    pub fn rng(mut self, rng: StdRng) -> Self {
        self.0.rng = rng;
        self
    }


}

impl<F: Float, L: Label> Default for RandomForestParams<F, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl <F: Float, L: Label> RandomForest<F, L> {
    /// Defaults are provided if the optional parameters are not specified:
    /// * `n_trees = 100`
    /// * `tree = DecisionTreeParams::new()`
    /// * `bootstrap_type = BootstrapType::BootstrapSamples(120)`

    #[allow(clippy::new_ret_no_self)]
    pub fn params() -> RandomForestParams<F, L> {
        RandomForestParams::new()
    }

}

impl<F: Float, L: Label> ParamGuard for RandomForestParams<F, L> {
    type Checked = RandomForestValidParams<F, L>;
    type Error = Error;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.n_trees == 0 {
            Err(Error::Parameters("Number of trees in a forest must be greater than zero!".to_string()))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
