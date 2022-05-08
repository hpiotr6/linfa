use std::fs::File;
use std::io::Write;

use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;

use linfa::prelude::*;
use linfa_forest::{Result};

fn main() -> Result<()> {
    // load Iris dataset
    let mut rng = SmallRng::seed_from_u64(42);



    Ok(())
}
