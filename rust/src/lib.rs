#![recursion_limit = "256"]

use pyo3::prelude::*;

mod array_utils;
mod ccallback;
mod column_grouping;
mod convolve2d;
mod correlate_nd;
mod fblas;
mod fftpack_convolve;
mod flapack;
mod fmm;
mod fpumode;
mod geometry;
mod givens_elimination;
mod graph_tools;
mod hausdorff;
mod kdtree;
mod levy_stable;
mod lfilter;
mod max_len_seq_inner;
mod median_filter_2d;
mod mio_utils;
mod ndimage_filters1d;
mod ndimage_label;
mod peak_finding_utils;
mod qmc;
mod qmvnt;
mod rank_filter_1d;
mod rcont;
mod remez;
mod reordering;
mod rgi;
mod sobol;
mod sosfilt;
mod spherical_voronoi;
mod spline;
mod upfirdn_apply;
mod vq;

#[cfg(test)]
mod testutil {
    /// Deterministic dependency-free PRNG (SplitMix64) for the unit tests.
    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in [-1, 1).
        pub fn f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 * (2.0 / (1u64 << 53) as f64) - 1.0
        }

        pub fn f32(&mut self) -> f32 {
            self.f64() as f32
        }
    }

    /// Max-norm of the elementwise difference of two equal-length vectors.
    pub fn max_abs_diff(x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y)
            .map(|(u, v)| (u - v).abs())
            .fold(0.0f64, f64::max)
    }

    /// Max absolute value.
    pub fn max_abs(v: &[f64]) -> f64 {
        v.iter().fold(0.0f64, |s, x| s.max(x.abs()))
    }
}

#[pymodule]
fn _scicore_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();

    let lib = PyModule::new(py, "_lib")?;
    ccallback::register(&lib)?;
    fpumode::register(&lib)?;
    module.add("_lib", lib)?;

    let linalg = PyModule::new(py, "linalg")?;
    array_utils::register(&linalg)?;
    fblas::register(&linalg)?;
    flapack::register(&linalg)?;
    givens_elimination::register(&linalg)?;
    module.add("linalg", linalg)?;

    let signal = PyModule::new(py, "signal")?;
    convolve2d::register(&signal)?;
    correlate_nd::register(&signal)?;
    lfilter::register(&signal)?;
    max_len_seq_inner::register(&signal)?;
    median_filter_2d::register(&signal)?;
    peak_finding_utils::register(&signal)?;
    remez::register(&signal)?;
    sosfilt::register(&signal)?;
    spline::register(&signal)?;
    upfirdn_apply::register(&signal)?;
    module.add("signal", signal)?;

    let fftpack = PyModule::new(py, "fftpack")?;
    fftpack_convolve::register(&fftpack)?;
    module.add("fftpack", fftpack)?;

    let cluster = PyModule::new(py, "cluster")?;
    vq::register(&cluster)?;
    module.add("cluster", cluster)?;

    let interpolate = PyModule::new(py, "interpolate")?;
    rgi::register(&interpolate)?;
    module.add("interpolate", interpolate)?;

    let csgraph = PyModule::new(py, "csgraph")?;
    reordering::register(&csgraph)?;
    graph_tools::register(&csgraph)?;
    module.add("csgraph", csgraph)?;

    let spatial = PyModule::new(py, "spatial")?;
    geometry::register(&spatial)?;
    hausdorff::register(&spatial)?;
    kdtree::register(&spatial)?;
    spherical_voronoi::register(&spatial)?;
    module.add("spatial", spatial)?;

    let stats = PyModule::new(py, "stats")?;
    rcont::register(&stats)?;
    levy_stable::register(&stats)?;
    qmvnt::register(&stats)?;
    qmc::register(&stats)?;
    sobol::register(&stats)?;
    module.add("stats", stats)?;

    let optimize = PyModule::new(py, "optimize")?;
    column_grouping::register(&optimize)?;
    module.add("optimize", optimize)?;

    let ndimage = PyModule::new(py, "ndimage")?;
    ndimage_label::register(&ndimage)?;
    ndimage_filters1d::register(&ndimage)?;
    rank_filter_1d::register(&ndimage)?;
    module.add("ndimage", ndimage)?;

    let io = PyModule::new(py, "io")?;
    mio_utils::register(&io)?;
    module.add("io", io)?;

    let fmm_mod = PyModule::new(py, "fmm")?;
    fmm::register(&fmm_mod)?;
    module.add("fmm", fmm_mod)?;

    Ok(())
}
