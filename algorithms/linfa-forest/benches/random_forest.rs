use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::prelude::*;
use ndarray::{concatenate, Array, Array1, Array2, Axis};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;
use rand::rngs::SmallRng;
