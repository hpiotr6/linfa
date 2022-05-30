//! Random Forest Classifier
//!


use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::collections::hash_set::Iter;
use std::hash::{Hash, Hasher};

use linfa::dataset::{AsMultiTargets, AsSingleTargets, AsTargets, FromTargetArray};
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Data, DataOwned, Dim, Dimension, Ix1, Ix2, OwnedRepr, ShapeBuilder};
use rand::rngs::SmallRng;
use rand::{Rng, RngCore, SeedableRng};

use super::{RandomForestValidParams, BootstrapType};
use linfa::{dataset::{Labels, Records}, error::Error, error::Result, traits::*, DatasetBase, Float, Label, ParamGuard, Dataset};
use linfa_trees::{DecisionTree, DecisionTreeParams, DecisionTreeValidParams};
use linfa::prelude::*;


#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};


pub struct RandomForest<F: Float, L: Label> {
    n_trees: usize,
    tree_params: DecisionTreeParams<F, L>,
    bootstrap_type: BootstrapType,
    trees: Vec<DecisionTree<F, L>>
}

fn get_bootstrap_iterator<'b, D, L: Label + Copy + 'b + std::fmt::Debug, F: Float, T, U: Clone + Copy, R: Rng, S>
(bootstrap_type: BootstrapType,
 dataset: &'b DatasetBase<ArrayBase<D, Ix2>, T>,
 rng: &'b mut R)
    // -> Box<dyn Iterator<Item=DatasetBase<Array2<F>, ArrayBase<OwnedRepr<L>, Dim<[usize; 1]>>>> + 'b>
        -> Box<dyn Iterator<Item=DatasetBase<ArrayBase<OwnedRepr<F>, Ix2>, ArrayBase<OwnedRepr<L>, Dim<[usize; 1]>>>> + 'b>
    where
        D: Data<Elem = F>,
        T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
        T: FromTargetArray<'b, Owned = ArrayBase<OwnedRepr<L>, Dim<[usize; 1]>>>,
        <T as FromTargetArray<'b>>::Owned: 'b,
        <T as linfa::dataset::FromTargetArray<'b>>::Owned: linfa::dataset::AsTargets
{
    // let records2 = dataset.records.to_owned();
    // let targets2= dataset.targets.as_targets().to_owned();
    // let dataset2 = DatasetBase::new(records2, targets2);
    match bootstrap_type {
        BootstrapType::BootstrapSamples(n_samples) =>
            Box::new(dataset.bootstrap_samples(n_samples, rng)),
        BootstrapType::BootstrapFeatures(n_features) =>
            Box::new(dataset.bootstrap_features(n_features, rng)),
        BootstrapType::BootstrapSamplesFeatures(n_samples, n_features) =>
            Box::new(dataset.bootstrap((n_samples, n_features), rng))
    }
}

// pub fn search<'b, F: Clone, E: Copy + 'b, T,
//     I: Iterator<Item = DatasetBase<Array2<F>, <T as FromTargetArray<'b>>::Owned>> + 'b>
// (t: I) -> bool
// where
//     T: FromTargetArray<'b, Elem = E>,
//     T::Owned: AsTargets
// {
//     t.any(|m| m == 1)
// }


impl<'a, F: Float, L: Label + 'a + std::fmt::Debug, D, T>
Fit<ArrayBase<D, Ix2>, T, Error>
for RandomForestValidParams<F, L>
    where
        D: Data<Elem = F>,
        T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
        T: FromTargetArray<'a>,
        <T as FromTargetArray<'a>>::Owned: AsTargets,
        <T as AsTargets>::Elem: Copy

{
    type Object = RandomForest<F, L>;


    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let mut rng = SmallRng::seed_from_u64(42); // TODO: rng jako parametr lasu
        // let x = dataset.bootstrap_samples(10, &mut rng);

        // TODO: czy da się to zrobić bez kopiowania całej bazy danych?
        let records2 = dataset.records.to_owned();
        let targets2= dataset.targets.as_targets().to_owned();
        let dataset2 = DatasetBase::new(records2, targets2);


        let mut subsample_iterator= match self.bootstrap_type() {
            BootstrapType::BootstrapSamples(n_samples) =>
                Box::new(dataset2.bootstrap_samples(n_samples, &mut rng))
                    as Box< dyn Iterator< Item = DatasetBase<Array2<F>, ArrayBase<OwnedRepr<_>, Dim<[usize; 1]>>> >>,
            BootstrapType::BootstrapFeatures(n_features) =>
                Box::new(dataset2.bootstrap_features(n_features, &mut rng))
                    as Box< dyn Iterator< Item = DatasetBase<Array2<F>, ArrayBase<OwnedRepr<_>, Dim<[usize; 1]>>> >>,
            BootstrapType::BootstrapSamplesFeatures(n_samples, n_features) =>
                Box::new(dataset2.bootstrap((n_samples, n_features), &mut rng))
                    as Box< dyn Iterator< Item = DatasetBase<Array2<F>, ArrayBase<OwnedRepr<_>, Dim<[usize; 1]>>> >>
        };

        let mut trees: Vec<DecisionTree<F, L>> = Vec::new();
        let tree_valid_params = self.tree_params().clone().check()?;


        // let mut subsample_iterator = get_bootstrap_iterator(self.bootstrap_type(), dataset, &mut rng);


        // let mut subsample_iterator = dataset2.bootstrap((120, 4), &mut rng);
        for _i in 0..self.n_trees() {
            let subsample = subsample_iterator.next().ok_or(std::fmt::Error).unwrap();
            trees.push(tree_valid_params.fit(&subsample)?);
        }

    Ok(RandomForest{
        n_trees: self.n_trees(),
        tree_params: self.tree_params().clone(),
        bootstrap_type: self.bootstrap_type(),
        trees
    })
     }
}

impl<F: Float, L: Label + Default, D: Data<Elem = F>>
PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
    for RandomForest<F, L> {

    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );
        println!("{:?}", x);
        println!("{:?}", y);
        make_prediction(x, y, self).unwrap();
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}


pub fn most_common<N: Eq + Label, D: Dimension>(
    array: &Array<N, D>
) -> Result<Array1<N>> where N: Hash{
    let mut ret: Vec<N> = Vec::new();
    println!("{:?}", array);
    for sample in array.columns() {
        let mut hashmap: HashMap<N, usize> = HashMap::new();
        for predict in sample {
            *hashmap.entry(predict.clone()).or_default() += 1;
        }
        ret.push(hashmap.into_iter().max_by_key(|cell| cell.1).ok_or("Empty vector?").unwrap().0);

    }
    Ok(Array1::from_vec(ret))
}


fn make_prediction<F: Float, L: Label + Default + Eq>(
    x : &ArrayBase<impl Data<Elem = F>, Ix2>,
    y : &mut Array1<L>,
    forest: &RandomForest<F, L>) -> Result<()>
{

    // klasyfikacja
    let mut predictions: Vec<L> = Vec::new();
    for (_i, tree) in forest.trees.iter().enumerate(){
        let mut gini_pred = tree.predict(x).to_vec();
        predictions.append(&mut gini_pred)
    }
    let predictions = Array2::from_shape_vec(
        (forest.n_trees, x.nsamples()), predictions)?;

    let final_prediction = most_common(&predictions)?;
    Ok(y.clone_from(&final_prediction))




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