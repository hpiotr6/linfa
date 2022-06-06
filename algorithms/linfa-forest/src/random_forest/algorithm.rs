//! Random Forest Classifier
//!

use std::collections::{HashMap};
use std::slice::Iter;
use linfa::dataset::{AsSingleTargets, AsTargets};
use ndarray::{Array, Array1, Array2, ArrayBase, Data, Ix2};

use super::{RandomForestValidParams, BootstrapType};
use linfa::{dataset::{Records}, error::Error, error::Result, traits::*, DatasetBase, Float, Label, ParamGuard};
use linfa_trees::{DecisionTree};

#[derive(Debug)]
pub struct RandomForest<F: Float, L: Label> {
    trees: Vec<DecisionTree<F, L>>,
}


impl<F: Float, L: Label + Copy, D, T> Fit<ArrayBase<D, Ix2>, T, Error>
for RandomForestValidParams<F, L>
    where
        D: Data<Elem = F>,
        T: AsSingleTargets<Elem = L>

{
    type Object = RandomForest<F, L>;

    /// Fit a decision tree using `hyperparameters` on the dataset consisting of
    /// a matrix of features `x` and an array of labels `y`.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {

        // cloning dataset for bootstrapping
        let dataset = DatasetBase::new(
            dataset.records.as_targets().to_owned(),
            dataset.targets.as_targets().to_owned())
            .with_weights(dataset.weights.as_targets().to_owned())
            .with_feature_names(dataset.feature_names());

        // creating bootstrapped dataset as infinite subsample-iterator
        let mut rng = self.rng().clone();
        let mut subsample_iterator: Box<dyn Iterator<Item = DatasetBase<Array2<F>, Array1<L>>>> =
            match self.bootstrap_type() {
            BootstrapType::BootstrapSamples(n_samples) =>
                Box::new(dataset.bootstrap_samples(n_samples, &mut rng)),
            BootstrapType::BootstrapFeatures(n_features) =>
                Box::new(dataset.bootstrap_features(n_features, &mut rng)),
            BootstrapType::BootstrapSamplesFeatures(n_samples, n_features) =>
                Box::new(dataset.bootstrap((n_samples, n_features), &mut rng))
        };

        let mut trees: Vec<DecisionTree<F, L>> = Vec::new();
        let tree_valid_params = self.tree_params().check()?;


        for _i in 0..self.n_trees() {
            let subsample = subsample_iterator.next().expect("Empty dataset?");
            trees.push(tree_valid_params.fit(&subsample)?);
        }

    Ok(RandomForest{
        trees
    })
     }
}

impl<F: Float, L: Label, D: Data<Elem = F>>
PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
    for RandomForest<F, L> {

    /// Make predictions for each row of a matrix of features `x`.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );
        make_prediction(x, y, self).unwrap();

    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}


/// Classify a sample &x based on majority of votes.
fn make_prediction<F: Float, L: Label, D: Data<Elem = F>>(
    x : &ArrayBase<D, Ix2>,
    y : &mut Array1<L>,
    forest: &RandomForest<F, L>) -> Result<()>
{

    // Make prediction for every single tree
    let mut trees_predictions: Vec<L> = Vec::new();
    for tree in forest.trees.iter() {
        trees_predictions.append(&mut tree.predict(x).to_vec());
    }
    let tree_predictions = Array2::from_shape_vec(
        (forest.trees.len(), x.nsamples()), trees_predictions)?;

    y.clone_from(&make_verdict(&tree_predictions)?);
    Ok(())
}

/// Verdict for final predictions based on decision trees voting.
pub fn make_verdict<L: Label> (trees_predictions: &Array<L, Ix2>) -> Result<Array1<L>>
{
    let mut verdict: Vec<L> = Vec::new();

    for sample in trees_predictions.columns() {
        let mut hashmap: HashMap<L, usize> = HashMap::new();
        for predict in sample {
            *hashmap.entry(predict.clone()).or_default() += 1;
        }
        verdict.push(hashmap.into_iter().max_by_key(|cell| cell.1).expect("Empty vector?").0);
    }

    Ok(Array1::from_vec(verdict))
}


impl<F: Float, L: Label> RandomForest<F, L> {

    pub fn iter_trees(&self) -> Iter<'_, DecisionTree<F, L>> {
        self.trees.iter()
    }

    pub fn trees(&self) -> &Vec<DecisionTree<F, L>> {
        &self.trees
    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use linfa::Dataset;

    #[test]
    fn test_most_common() -> Result<()> {
        let predictions = array![[0, 1, 2, 3],
                                              [1, 1, 0, 3],
                                              [1, 1, 0, 2],
                                              [1, 1, 2, 2],
                                              [0, 1, 2, 3]];
        let result = array![1,1,2,3];
        let predicted_winners = make_verdict(&predictions);
        println!("Our winner: {:?}", predicted_winners);
        println!("Our result: {:?}", result);
        assert_eq!(&result, &predicted_winners?);
        Ok(())
    }



    #[test]
    /// Check if model trained on same data as tested gives proper target
    fn simple_case() -> Result<()> {
        let data = array![[1., 2., 3.], [1., 2., 4.], [1., 3., 3.5]];
        let targets = array![0, 0, 1];

        let dataset = Dataset::new(data.clone(), targets);
        let model = RandomForest::params().fit(&dataset)?;

        assert_eq!(model.predict(&data), array![0, 0, 1]);

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

    // #[test]
    // /// Check that for random data the n trees is used
    // fn check_n_trees() -> Result<()> {
    //     let mut rng = SmallRng::seed_from_u64(42);
    //
    //     // create very sparse data
    //     let data = Array::random_using((50, 50), Uniform::new(-1., 1.), &mut rng);
    //     let targets = (0..50).collect::<Array1<usize>>();
    //     //
    //     let dataset = Dataset::new(data, targets);
    //     //
    //     // // check that the provided depth is actually used
    //     // for max_depth in &[1, 5, 10, 20] {
    //     //     let model = DecisionTree::params()
    //     //         .max_depth(Some(*max_depth))
    //     //         .min_impurity_decrease(1e-10f64)
    //     //         .min_weight_split(1e-10)
    //     //         .fit(&dataset)?;
    //     //     assert_eq!(model.max_depth(), *max_depth);
    //     // }
    //     for trees_num in [1, 5, 20, 100]{
    //         let model = RandomForest::params().n_trees(trees_num).fit(&dataset)?;
    //         assert_eq!(model.n_trees, trees_num);
    //
    //     }
    //
    //     Ok(())
    // }
}