//! kd-tree (cKDTree) — Rust port of scipy's C++ `ckdtree`.
//!
//! The heavy computation (tree construction + nearest-neighbor / ball
//! queries) lives here. The Python `cKDTree` wrapper in
//! `scicore/spatial/_ckdtree.py` handles scipy-compatible output formatting
//! (squeezing, list-of-lists, sets, sparse matrices).
//!
//! Distances are tracked internally in `d**p` units (squared for `p=2`,
//! the raw value for `p=1`/`p=inf`, `d**p` for general `p`), matching the
//! C++ reference implementation.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::BinaryHeap;

/// A total-order wrapper around `f64` so it can be used in a `BinaryHeap`.
/// `f64` only implements `PartialOrd`; `total_cmp` gives a total order
/// (NaN sorts after all finite values and `+inf`).
#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ===========================================================================
// Distance helpers
// ===========================================================================

/// `distance**p` between two `m`-dim points, early-exiting once it exceeds `ub`.
fn point_point_p(a: &[f64], b: &[f64], p: f64, ub: f64) -> f64 {
    let m = a.len();
    if p == 2.0 {
        let mut s = 0.0;
        for k in 0..m {
            let d = a[k] - b[k];
            s += d * d;
            if s > ub {
                return s;
            }
        }
        s
    } else if p == 1.0 {
        let mut s = 0.0;
        for k in 0..m {
            s += (a[k] - b[k]).abs();
            if s > ub {
                return s;
            }
        }
        s
    } else if p.is_infinite() {
        let mut s = 0.0;
        for k in 0..m {
            let d = (a[k] - b[k]).abs();
            if d > s {
                s = d;
            }
            if s > ub {
                return s;
            }
        }
        s
    } else {
        let mut s = 0.0;
        for k in 0..m {
            s += (a[k] - b[k]).abs().powf(p);
            if s > ub {
                return s;
            }
        }
        s
    }
}

/// Per-dimension `(gap, extent)` between two intervals.
fn interval_gap_extent(a_min: f64, a_max: f64, b_min: f64, b_max: f64) -> (f64, f64) {
    let gap = (a_min - b_max).max(b_min - a_max).max(0.0);
    let ext = (a_max - b_min).max(b_max - a_min);
    (gap, ext)
}

/// Combined `(min, max)` distance in `d**p` units between two `m`-dim boxes.
fn box_box_min_max(
    m: usize,
    r1_min: &[f64],
    r1_max: &[f64],
    r2_min: &[f64],
    r2_max: &[f64],
    p: f64,
) -> (f64, f64) {
    let mut min_d: f64 = 0.0;
    let mut max_d: f64 = 0.0;
    for k in 0..m {
        let (gap, ext) = interval_gap_extent(r1_min[k], r1_max[k], r2_min[k], r2_max[k]);
        if p.is_infinite() {
            min_d = min_d.max(gap);
            max_d = max_d.max(ext);
        } else if p == 1.0 {
            min_d += gap;
            max_d += ext;
        } else {
            min_d += gap.powf(p);
            max_d += ext.powf(p);
        }
    }
    (min_d, max_d)
}

/// Convert an internal `d**p` value to a true distance.
fn from_pow(d: f64, p: f64) -> f64 {
    if p == 2.0 {
        d.sqrt()
    } else if p == 1.0 || p.is_infinite() {
        d
    } else {
        d.powf(1.0 / p)
    }
}

/// eps approximation factor in `d**p` units.
fn epsfac(eps: f64, p: f64) -> f64 {
    if eps == 0.0 {
        1.0
    } else if p.is_infinite() {
        1.0 / (1.0 + eps)
    } else {
        1.0 / (1.0 + eps).powf(p)
    }
}

/// Raise a true distance bound to `d**p` units.
fn to_pow(d: f64, p: f64) -> f64 {
    if p.is_infinite() { d } else { d.powf(p) }
}

/// The next representable `f64` after `x` toward `+inf` (one ULP up).
fn nextafter_up(x: f64) -> f64 {
    if x == f64::INFINITY {
        return x;
    }
    let bits = x.to_bits();
    let next_bits = if x >= 0.0 { bits + 1 } else { bits - 1 };
    f64::from_bits(next_bits)
}

// ===========================================================================
// Tree construction helpers
// ===========================================================================

/// The `k`-th smallest value in `values` (0-indexed), via quickselect.
fn kth_smallest(values: &[f64], k: usize) -> f64 {
    let mut v = values.to_vec();
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let k = k.min(n - 1);
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        v.swap(mid, hi - 1);
        let pivot = v[hi - 1];
        let mut i = lo;
        for j in lo..hi - 1 {
            if v[j] < pivot {
                v.swap(i, j);
                i += 1;
            }
        }
        v.swap(i, hi - 1);
        if k == i {
            break;
        } else if k < i {
            hi = i;
        } else {
            lo = i + 1;
        }
    }
    v[k]
}

/// Partition `indices` in place so that elements with `data[idx*m+d] < pivot`
/// come first; returns the count of such elements.
fn partition_around(data: &[f64], m: usize, d: usize, indices: &mut [usize], pivot: f64) -> usize {
    let mut i = 0usize;
    for j in 0..indices.len() {
        if data[indices[j] * m + d] < pivot {
            indices.swap(i, j);
            i += 1;
        }
    }
    i
}

/// Push the ordered pair `(min(i,j), max(i,j))`.
fn push_ordered_pair(pairs: &mut Vec<(usize, usize)>, i: usize, j: usize) {
    if i < j {
        pairs.push((i, j));
    } else {
        pairs.push((j, i));
    }
}

/// First index `k` in ascending `v` with `v[k] >= x` (upper count).
fn lower_bound(v: &[f64], x: f64) -> usize {
    v.iter().position(|&y| y >= x).unwrap_or(v.len())
}

// ===========================================================================
// Array extraction (buffer fast path + tolist fallback)
// ===========================================================================

fn flatten_f64_into(v: &Bound<'_, PyAny>, out: &mut Vec<f64>) -> PyResult<()> {
    if let Ok(list) = v.cast::<PyList>() {
        for item in list.iter() {
            flatten_f64_into(&item, out)?;
        }
    } else {
        out.push(v.extract::<f64>()?);
    }
    Ok(())
}

/// Extract a flat row-major `Vec<f64>` from an rsnumpy array (or nested list).
fn extract_f64_flat(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(buf) = PyBuffer::<f64>::get(obj) {
        let n = buf.item_count();
        return Ok(unsafe { std::slice::from_raw_parts(buf.buf_ptr() as *const f64, n) }.to_vec());
    }
    let mut out = Vec::new();
    flatten_f64_into(obj, &mut out)?;
    Ok(out)
}

// ===========================================================================
// kd-tree data structure
// ===========================================================================

#[derive(Clone, Copy, Default)]
struct Node {
    start_idx: usize,
    end_idx: usize,
    children: usize,
    split_dim: i32, // -1 = leaf
    split: f64,
    less: usize,
    greater: usize,
}

impl Node {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.split_dim == -1
    }
}

struct KdTree {
    m: usize,
    n: usize,
    raw_data: Vec<f64>,      // original order, row-major n*m
    raw_indices: Vec<usize>, // permutation (tree in-order)
    node_mins: Vec<f64>,     // flat: node_index*m .. +m
    node_maxes: Vec<f64>,    // flat: node_index*m .. +m
    nodes: Vec<Node>,        // flat, root = nodes[0]
    leafsize: usize,
    // Accepted for API compatibility; periodic (toroidal) distance is not
    // implemented yet, so this is stored but not (yet) used in the metric.
    #[allow(dead_code)]
    boxsize: Option<Vec<f64>>,
}

impl KdTree {
    fn new(
        raw_data: Vec<f64>,
        m: usize,
        n: usize,
        leafsize: usize,
        compact: bool,
        median: bool,
        boxsize: Option<Vec<f64>>,
    ) -> Self {
        let raw_indices: Vec<usize> = (0..n).collect();
        let nodes: Vec<Node> = Vec::new();
        let node_mins: Vec<f64> = Vec::new();
        let node_maxes: Vec<f64> = Vec::new();

        // global bounds
        let mut mins = vec![f64::INFINITY; m];
        let mut maxes = vec![f64::NEG_INFINITY; m];
        for i in 0..n {
            for k in 0..m {
                let v = raw_data[i * m + k];
                if v < mins[k] {
                    mins[k] = v;
                }
                if v > maxes[k] {
                    maxes[k] = v;
                }
            }
        }
        if n == 0 {
            for k in 0..m {
                mins[k] = 0.0;
                maxes[k] = 0.0;
            }
        }

        let mut tree = KdTree {
            m,
            n,
            raw_data,
            raw_indices,
            node_mins,
            node_maxes,
            nodes,
            leafsize: leafsize.max(1),
            boxsize,
        };
        if n > 0 {
            tree.build(0, n, &mut maxes, &mut mins, median, compact);
        } else {
            tree.nodes.push(Node {
                start_idx: 0,
                end_idx: 0,
                children: 0,
                split_dim: -1,
                split: 0.0,
                less: 0,
                greater: 0,
            });
        }
        tree
    }

    /// Build the subtree over `raw_indices[start_idx..end_idx)`; returns the node index.
    fn build(
        &mut self,
        start_idx: usize,
        end_idx: usize,
        maxes: &mut [f64],
        mins: &mut [f64],
        median: bool,
        compact: bool,
    ) -> usize {
        self.nodes.push(Node::default());
        let node_index = self.nodes.len() - 1;
        self.nodes[node_index].start_idx = start_idx;
        self.nodes[node_index].end_idx = end_idx;
        self.nodes[node_index].children = end_idx - start_idx;

        let m = self.m;
        if end_idx - start_idx <= self.leafsize {
            // leaf: use the passed-in bounds
            self.node_mins.extend_from_slice(mins);
            self.node_maxes.extend_from_slice(maxes);
            self.nodes[node_index].split_dim = -1;
            return node_index;
        }

        // inner node: recompute tight bounds if compact
        if compact {
            let first = self.raw_indices[start_idx] * m;
            maxes.copy_from_slice(&self.raw_data[first..first + m]);
            mins.copy_from_slice(&self.raw_data[first..first + m]);
            for j in (start_idx + 1)..end_idx {
                let base = self.raw_indices[j] * m;
                for i in 0..m {
                    let v = self.raw_data[base + i];
                    if v > maxes[i] {
                        maxes[i] = v;
                    }
                    if v < mins[i] {
                        mins[i] = v;
                    }
                }
            }
        }
        self.node_mins.extend_from_slice(mins);
        self.node_maxes.extend_from_slice(maxes);

        // split on the dimension with the largest spread
        let mut d = 0usize;
        let mut size = 0.0f64;
        for i in 0..m {
            let s = maxes[i] - mins[i];
            if s > size {
                d = i;
                size = s;
            }
        }
        let maxval = maxes[d];
        let minval = mins[d];
        if maxval == minval {
            // all points identical in this dimension -> leaf
            self.nodes[node_index].split_dim = -1;
            return node_index;
        }

        // split value
        let split = if median {
            let count = end_idx - start_idx;
            let mut vals = Vec::with_capacity(count);
            for j in start_idx..end_idx {
                vals.push(self.raw_data[self.raw_indices[j] * m + d]);
            }
            kth_smallest(&vals, count / 2)
        } else {
            (maxval + minval) / 2.0
        };

        // partition raw_indices[start_idx..end_idx) around `split`
        let mut p = {
            let data = &self.raw_data;
            let sub = &mut self.raw_indices[start_idx..end_idx];
            start_idx + partition_around(data, m, d, sub, split)
        };

        // slide the midpoint if the partition is degenerate
        if p == start_idx {
            let min_v = self.raw_indices[start_idx..end_idx]
                .iter()
                .map(|&idx| self.raw_data[idx * m + d])
                .fold(f64::INFINITY, f64::min);
            let split2 = nextafter_up(min_v);
            p = {
                let data = &self.raw_data;
                let sub = &mut self.raw_indices[start_idx..end_idx];
                start_idx + partition_around(data, m, d, sub, split2)
            };
        } else if p == end_idx {
            let max_v = self.raw_indices[start_idx..end_idx]
                .iter()
                .map(|&idx| self.raw_data[idx * m + d])
                .fold(f64::NEG_INFINITY, f64::max);
            p = {
                let data = &self.raw_data;
                let sub = &mut self.raw_indices[start_idx..end_idx];
                start_idx + partition_around(data, m, d, sub, max_v)
            };
        }

        // fallback: if still degenerate, make this node a leaf
        if p == start_idx || p == end_idx {
            self.nodes[node_index].split_dim = -1;
            return node_index;
        }

        // child bounds
        let (mut less_mins, mut less_maxes, mut greater_mins, mut greater_maxes) = if compact {
            (mins.to_vec(), maxes.to_vec(), mins.to_vec(), maxes.to_vec())
        } else {
            let mut lm = maxes.to_vec();
            lm[d] = split;
            let mut gm = mins.to_vec();
            gm[d] = split;
            (mins.to_vec(), lm, mins.to_vec(), gm)
        };

        let less = self.build(
            start_idx,
            p,
            &mut less_maxes,
            &mut less_mins,
            median,
            compact,
        );
        let greater = self.build(
            p,
            end_idx,
            &mut greater_maxes,
            &mut greater_mins,
            median,
            compact,
        );

        self.nodes[node_index].less = less;
        self.nodes[node_index].greater = greater;
        self.nodes[node_index].split_dim = d as i32;
        self.nodes[node_index].split = split;
        node_index
    }

    #[inline]
    fn node_box(&self, node_index: usize) -> (&[f64], &[f64]) {
        let m = self.m;
        let s = node_index * m;
        (&self.node_mins[s..s + m], &self.node_maxes[s..s + m])
    }

    // -----------------------------------------------------------------------
    // k-nearest-neighbor query (best-first search)
    // -----------------------------------------------------------------------

    fn query_single_point(
        &self,
        x: &[f64],
        klist: &[usize],
        eps: f64,
        p: f64,
        dub_pow: f64,
    ) -> (Vec<f64>, Vec<usize>) {
        let m = self.m;
        let kmax = klist.iter().copied().max().unwrap_or(0);
        let ef = epsfac(eps, p);

        let mut x_min = vec![0.0f64; m];
        let mut x_max = vec![0.0f64; m];
        x_min.copy_from_slice(x);
        x_max.copy_from_slice(x);

        let mut best_dub = dub_pow;
        let mut neighbors: BinaryHeap<(OrdF64, usize)> = BinaryHeap::with_capacity(kmax.max(1));
        // min-distance-first node queue: store (-min_dist, node) in a max-heap
        let mut queue: BinaryHeap<(OrdF64, usize)> = BinaryHeap::new();
        queue.push((OrdF64(0.0), 0));

        while let Some((neg_md, node_idx)) = queue.pop() {
            let md = -neg_md.0;
            if md > best_dub * ef {
                break;
            }
            let node = &self.nodes[node_idx];
            if node.is_leaf() {
                for i in node.start_idx..node.end_idx {
                    let idx = self.raw_indices[i];
                    let base = idx * m;
                    let d = point_point_p(&self.raw_data[base..base + m], x, p, best_dub);
                    if d < best_dub {
                        if neighbors.len() >= kmax {
                            neighbors.pop();
                        }
                        neighbors.push((OrdF64(d), idx));
                        if neighbors.len() == kmax {
                            best_dub = neighbors.peek().map(|t| t.0.0).unwrap_or(best_dub);
                        }
                    }
                }
            } else {
                for &child in &[node.less, node.greater] {
                    let (cmin, cmax) = self.node_box(child);
                    let (min_d, _) = box_box_min_max(m, &x_min, &x_max, cmin, cmax, p);
                    if min_d <= best_dub * ef {
                        queue.push((OrdF64(-min_d), child));
                    }
                }
            }
        }

        let mut sorted: Vec<(f64, usize)> = neighbors.into_iter().map(|(d, i)| (d.0, i)).collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut dists = Vec::with_capacity(klist.len());
        let mut ids = Vec::with_capacity(klist.len());
        for &k in klist {
            if k > sorted.len() {
                dists.push(f64::INFINITY);
                ids.push(self.n);
            } else {
                dists.push(from_pow(sorted[k - 1].0, p));
                ids.push(sorted[k - 1].1);
            }
        }
        (dists, ids)
    }

    fn query(
        &self,
        x_flat: &[f64],
        klist: &[usize],
        eps: f64,
        p: f64,
        dub_pow: f64,
    ) -> (Vec<f64>, Vec<usize>) {
        let m = self.m;
        let nq = x_flat.len() / m.max(1);
        let nk = klist.len();
        let mut d_flat = Vec::with_capacity(nq * nk);
        let mut i_flat = Vec::with_capacity(nq * nk);
        for q in 0..nq {
            let (d, i) =
                self.query_single_point(&x_flat[q * m..(q + 1) * m], klist, eps, p, dub_pow);
            d_flat.extend(d);
            i_flat.extend(i);
        }
        (d_flat, i_flat)
    }

    // -----------------------------------------------------------------------
    // Ball queries (recursive DFS with bounding-box pruning)
    // -----------------------------------------------------------------------

    /// Collect all indices in `self` within `upper_bound` (`d**p`) of point `x`.
    #[allow(clippy::too_many_arguments)]
    fn query_ball_point_dfs(
        &self,
        x: &[f64],
        x_min: &[f64],
        x_max: &[f64],
        node: usize,
        upper_bound: f64,
        ef: f64,
        p: f64,
        out: &mut Vec<usize>,
    ) {
        let m = self.m;
        let (nmin, nmax) = self.node_box(node);
        let (min_d, max_d) = box_box_min_max(m, x_min, x_max, nmin, nmax, p);
        if min_d > upper_bound * ef {
            return;
        }
        let n = &self.nodes[node];
        if max_d < upper_bound / ef {
            for i in n.start_idx..n.end_idx {
                out.push(self.raw_indices[i]);
            }
            return;
        }
        if n.is_leaf() {
            for i in n.start_idx..n.end_idx {
                let idx = self.raw_indices[i];
                let base = idx * m;
                let d = point_point_p(&self.raw_data[base..base + m], x, p, upper_bound);
                if d <= upper_bound {
                    out.push(idx);
                }
            }
        } else {
            self.query_ball_point_dfs(x, x_min, x_max, n.less, upper_bound, ef, p, out);
            self.query_ball_point_dfs(x, x_min, x_max, n.greater, upper_bound, ef, p, out);
        }
    }

    /// For each point in `self`, collect the indices in `other` within `upper_bound`.
    #[allow(clippy::too_many_arguments)]
    fn query_ball_tree_dfs(
        &self,
        other: &KdTree,
        n1: usize,
        n2: usize,
        upper_bound: f64,
        ef: f64,
        p: f64,
        results: &mut Vec<Vec<usize>>,
    ) {
        let m = self.m;
        let (a_min, a_max) = self.node_box(n1);
        let (b_min, b_max) = other.node_box(n2);
        let (min_d, max_d) = box_box_min_max(m, a_min, a_max, b_min, b_max, p);
        if min_d > upper_bound * ef {
            return;
        }
        let node1 = &self.nodes[n1];
        let node2 = &other.nodes[n2];
        let less1 = node1.less;
        let greater1 = node1.greater;
        let less2 = node2.less;
        let greater2 = node2.greater;
        if max_d < upper_bound / ef {
            for i in node1.start_idx..node1.end_idx {
                let si = self.raw_indices[i];
                for j in node2.start_idx..node2.end_idx {
                    results[si].push(other.raw_indices[j]);
                }
            }
            return;
        }
        if node1.is_leaf() && node2.is_leaf() {
            for i in node1.start_idx..node1.end_idx {
                let si = self.raw_indices[i];
                let sbase = si * m;
                for j in node2.start_idx..node2.end_idx {
                    let oj = other.raw_indices[j];
                    let obase = oj * m;
                    let d = point_point_p(
                        &self.raw_data[sbase..sbase + m],
                        &other.raw_data[obase..obase + m],
                        p,
                        upper_bound,
                    );
                    if d <= upper_bound {
                        results[si].push(oj);
                    }
                }
            }
        } else if !node1.is_leaf() && node2.is_leaf() {
            self.query_ball_tree_dfs(other, less1, n2, upper_bound, ef, p, results);
            self.query_ball_tree_dfs(other, greater1, n2, upper_bound, ef, p, results);
        } else if node1.is_leaf() && !node2.is_leaf() {
            self.query_ball_tree_dfs(other, n1, less2, upper_bound, ef, p, results);
            self.query_ball_tree_dfs(other, n1, greater2, upper_bound, ef, p, results);
        } else {
            self.query_ball_tree_dfs(other, less1, less2, upper_bound, ef, p, results);
            self.query_ball_tree_dfs(other, less1, greater2, upper_bound, ef, p, results);
            self.query_ball_tree_dfs(other, greater1, less2, upper_bound, ef, p, results);
            self.query_ball_tree_dfs(other, greater1, greater2, upper_bound, ef, p, results);
        }
    }

    /// Collect all pairs `(i, j)` with `i < j` within `upper_bound` in `self`.
    fn query_pairs_dfs(
        &self,
        n1: usize,
        n2: usize,
        upper_bound: f64,
        ef: f64,
        p: f64,
        pairs: &mut Vec<(usize, usize)>,
    ) {
        let m = self.m;
        let (a_min, a_max) = self.node_box(n1);
        let (b_min, b_max) = self.node_box(n2);
        let (min_d, max_d) = box_box_min_max(m, a_min, a_max, b_min, b_max, p);
        if min_d > upper_bound * ef {
            return;
        }
        let node1 = &self.nodes[n1];
        let node2 = &self.nodes[n2];
        let same = n1 == n2;
        let leaf_ub = upper_bound / ef;
        let less1 = node1.less;
        let greater1 = node1.greater;
        let less2 = node2.less;
        let greater2 = node2.greater;
        if max_d < upper_bound / ef {
            if same {
                for i in node1.start_idx..node1.end_idx {
                    let si = self.raw_indices[i];
                    for j in (i + 1)..node1.end_idx {
                        push_ordered_pair(pairs, si, self.raw_indices[j]);
                    }
                }
            } else {
                for i in node1.start_idx..node1.end_idx {
                    let si = self.raw_indices[i];
                    for j in node2.start_idx..node2.end_idx {
                        push_ordered_pair(pairs, si, self.raw_indices[j]);
                    }
                }
            }
            return;
        }
        if node1.is_leaf() && node2.is_leaf() {
            for i in node1.start_idx..node1.end_idx {
                let si = self.raw_indices[i];
                let sbase = si * m;
                let i_start = if same { i + 1 } else { node2.start_idx };
                for j in i_start..node2.end_idx {
                    let sj = self.raw_indices[j];
                    let jbase = sj * m;
                    let d = point_point_p(
                        &self.raw_data[sbase..sbase + m],
                        &self.raw_data[jbase..jbase + m],
                        p,
                        leaf_ub,
                    );
                    if d <= leaf_ub {
                        push_ordered_pair(pairs, si, sj);
                    }
                }
            }
        } else if !node1.is_leaf() && node2.is_leaf() {
            self.query_pairs_dfs(less1, n2, upper_bound, ef, p, pairs);
            self.query_pairs_dfs(greater1, n2, upper_bound, ef, p, pairs);
        } else if node1.is_leaf() && !node2.is_leaf() {
            self.query_pairs_dfs(n1, less2, upper_bound, ef, p, pairs);
            self.query_pairs_dfs(n1, greater2, upper_bound, ef, p, pairs);
        } else {
            self.query_pairs_dfs(less1, less2, upper_bound, ef, p, pairs);
            self.query_pairs_dfs(less1, greater2, upper_bound, ef, p, pairs);
            if !same {
                self.query_pairs_dfs(greater1, less2, upper_bound, ef, p, pairs);
            }
            self.query_pairs_dfs(greater1, greater2, upper_bound, ef, p, pairs);
        }
    }

    /// Collect `(i, j, true_distance)` for all pairs within `upper_bound` (`d**p`).
    fn sparse_distance_matrix_dfs(
        &self,
        other: &KdTree,
        n1: usize,
        n2: usize,
        upper_bound: f64,
        p: f64,
        out: &mut Vec<(usize, usize, f64)>,
    ) {
        let m = self.m;
        let (a_min, a_max) = self.node_box(n1);
        let (b_min, b_max) = other.node_box(n2);
        let (min_d, _) = box_box_min_max(m, a_min, a_max, b_min, b_max, p);
        if min_d > upper_bound {
            return;
        }
        let node1 = &self.nodes[n1];
        let node2 = &other.nodes[n2];
        let less1 = node1.less;
        let greater1 = node1.greater;
        let less2 = node2.less;
        let greater2 = node2.greater;
        if node1.is_leaf() && node2.is_leaf() {
            for i in node1.start_idx..node1.end_idx {
                let si = self.raw_indices[i];
                let sbase = si * m;
                for j in node2.start_idx..node2.end_idx {
                    let oj = other.raw_indices[j];
                    let obase = oj * m;
                    let d = point_point_p(
                        &self.raw_data[sbase..sbase + m],
                        &other.raw_data[obase..obase + m],
                        p,
                        upper_bound,
                    );
                    if d <= upper_bound {
                        out.push((si, oj, from_pow(d, p)));
                    }
                }
            }
        } else if !node1.is_leaf() && node2.is_leaf() {
            self.sparse_distance_matrix_dfs(other, less1, n2, upper_bound, p, out);
            self.sparse_distance_matrix_dfs(other, greater1, n2, upper_bound, p, out);
        } else if node1.is_leaf() && !node2.is_leaf() {
            self.sparse_distance_matrix_dfs(other, n1, less2, upper_bound, p, out);
            self.sparse_distance_matrix_dfs(other, n1, greater2, upper_bound, p, out);
        } else {
            self.sparse_distance_matrix_dfs(other, less1, less2, upper_bound, p, out);
            self.sparse_distance_matrix_dfs(other, less1, greater2, upper_bound, p, out);
            self.sparse_distance_matrix_dfs(other, greater1, less2, upper_bound, p, out);
            self.sparse_distance_matrix_dfs(other, greater1, greater2, upper_bound, p, out);
        }
    }

    /// Weight of the subtree rooted at `node`: the per-node weight sum if weights
    /// are given, else the number of points in the subtree.
    fn node_weight(
        &self,
        node: usize,
        weights: Option<&[f64]>,
        node_weights: Option<&[f64]>,
    ) -> f64 {
        match (weights, node_weights) {
            (Some(_), Some(nw)) => nw[node],
            _ => self.nodes[node].children as f64,
        }
    }

    /// Per-node sums of point `weights` (children accumulate into parents, pre-order).
    fn node_weight_sum(&self, weights: &[f64]) -> Vec<f64> {
        let mut sums = vec![0.0f64; self.nodes.len()];
        for ni in (0..self.nodes.len()).rev() {
            let node = &self.nodes[ni];
            if node.is_leaf() {
                let mut s = 0.0f64;
                for i in node.start_idx..node.end_idx {
                    s += weights[self.raw_indices[i]];
                }
                sums[ni] = s;
            } else {
                sums[ni] = sums[node.less] + sums[node.greater];
            }
        }
        sums
    }

    /// Count neighbor pairs binned by radius. `real_r` must be sorted ascending and
    /// already in `d**p` units. `results` has `real_r.len()` entries, zeroed.
    #[allow(clippy::too_many_arguments)]
    fn count_neighbors_dfs(
        &self,
        other: &KdTree,
        n1: usize,
        n2: usize,
        real_r: &[f64],
        p: f64,
        sw: Option<&[f64]>,
        ow: Option<&[f64]>,
        snw: Option<&[f64]>,
        onw: Option<&[f64]>,
        cumulative: bool,
        results: &mut [f64],
    ) {
        let m = self.m;
        let (a_min, a_max) = self.node_box(n1);
        let (b_min, b_max) = other.node_box(n2);
        let (min_d, max_d) = box_box_min_max(m, a_min, a_max, b_min, b_max, p);

        let new_start = lower_bound(real_r, min_d);
        let new_end = lower_bound(real_r, max_d);
        let w1 = self.node_weight(n1, sw, snw);
        let w2 = other.node_weight(n2, ow, onw);

        if cumulative {
            if new_end < real_r.len() {
                let nn = w1 * w2;
                for r in &mut results[new_end..] {
                    *r += nn;
                }
            }
        } else if new_end == new_start {
            results[new_start] += w1 * w2;
        }

        if new_end == new_start {
            return;
        }
        let node1 = &self.nodes[n1];
        let node2 = &other.nodes[n2];
        if node1.is_leaf() && node2.is_leaf() {
            for i in node1.start_idx..node1.end_idx {
                let si = self.raw_indices[i];
                let sbase = si * m;
                let wsi = match sw {
                    Some(w) => w[si],
                    None => 1.0,
                };
                for j in node2.start_idx..node2.end_idx {
                    let oj = other.raw_indices[j];
                    let obase = oj * m;
                    let wo = match ow {
                        Some(w) => w[oj],
                        None => 1.0,
                    };
                    let d = point_point_p(
                        &self.raw_data[sbase..sbase + m],
                        &other.raw_data[obase..obase + m],
                        p,
                        f64::INFINITY,
                    );
                    if cumulative {
                        for l in new_start..new_end {
                            if d <= real_r[l] {
                                results[l] += wsi * wo;
                            }
                        }
                    } else {
                        let l = lower_bound(&real_r[new_start..new_end], d);
                        if l < new_end - new_start {
                            results[new_start + l] += wsi * wo;
                        }
                    }
                }
            }
            return;
        }
        let less1 = node1.less;
        let greater1 = node1.greater;
        let less2 = node2.less;
        let greater2 = node2.greater;
        if node1.is_leaf() && !node2.is_leaf() {
            self.count_neighbors_dfs(
                other, n1, less2, real_r, p, sw, ow, snw, onw, cumulative, results,
            );
            self.count_neighbors_dfs(
                other, n1, greater2, real_r, p, sw, ow, snw, onw, cumulative, results,
            );
        } else if !node1.is_leaf() && node2.is_leaf() {
            self.count_neighbors_dfs(
                other, less1, n2, real_r, p, sw, ow, snw, onw, cumulative, results,
            );
            self.count_neighbors_dfs(
                other, greater1, n2, real_r, p, sw, ow, snw, onw, cumulative, results,
            );
        } else {
            self.count_neighbors_dfs(
                other, less1, less2, real_r, p, sw, ow, snw, onw, cumulative, results,
            );
            self.count_neighbors_dfs(
                other, less1, greater2, real_r, p, sw, ow, snw, onw, cumulative, results,
            );
            self.count_neighbors_dfs(
                other, greater1, less2, real_r, p, sw, ow, snw, onw, cumulative, results,
            );
            self.count_neighbors_dfs(
                other, greater1, greater2, real_r, p, sw, ow, snw, onw, cumulative, results,
            );
        }
    }
}

// ===========================================================================
// pyo3 binding
// ===========================================================================

#[pyclass(name = "KDTreeCore")]
struct KdTreeCore {
    inner: KdTree,
}

#[pymethods]
impl KdTreeCore {
    #[new]
    fn new(
        data: &Bound<'_, PyAny>,
        leafsize: usize,
        compact_nodes: bool,
        _copy_data: bool,
        balanced_tree: bool,
        boxsize: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let flat = extract_f64_flat(data)?;
        let shape_obj = data.getattr("shape")?;
        let shape: Vec<usize> = shape_obj.extract()?;
        let (n, m) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1usize, shape[0])
        } else if flat.is_empty() {
            (0usize, 0usize)
        } else {
            (1usize, flat.len())
        };
        if n > 0 && flat.len() != n * m {
            return Err(PyValueError::new_err(
                "data length does not match its shape",
            ));
        }
        let bs = if boxsize.is_none() {
            None
        } else {
            Some(extract_f64_flat(boxsize)?)
        };
        let inner = KdTree::new(flat, m, n, leafsize, compact_nodes, balanced_tree, bs);
        Ok(KdTreeCore { inner })
    }

    /// k-nearest neighbors. `k` is a list of 1-based ranks; `x` is a flat
    /// row-major `(nq, m)` array. Returns flat `(distances, indices)` of
    /// length `nq * len(k)`.
    #[pyo3(name = "query")]
    fn query_impl(
        &self,
        x: &Bound<'_, PyAny>,
        k: Vec<usize>,
        eps: f64,
        p: f64,
        distance_upper_bound: f64,
    ) -> PyResult<(Vec<f64>, Vec<usize>)> {
        let x_flat = extract_f64_flat(x)?;
        let dub_pow = to_pow(distance_upper_bound, p);
        Ok(self.inner.query(&x_flat, &k, eps, p, dub_pow))
    }

    #[getter]
    fn m(&self) -> usize {
        self.inner.m
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    /// Per-dimension max of the data (the root node's box = global bounds).
    fn maxes(&self) -> Vec<f64> {
        if self.inner.nodes.is_empty() {
            return vec![f64::NAN; self.inner.m];
        }
        self.inner.node_maxes[0..self.inner.m].to_vec()
    }

    /// Per-dimension min of the data (the root node's box = global bounds).
    fn mins(&self) -> Vec<f64> {
        if self.inner.nodes.is_empty() {
            return vec![f64::NAN; self.inner.m];
        }
        self.inner.node_mins[0..self.inner.m].to_vec()
    }

    /// Node structure: `(split_dim, split, children, is_leaf, less, greater, start, end)`.
    fn node_info(&self, node_index: usize) -> (i32, f64, usize, bool, usize, usize, usize, usize) {
        let n = &self.inner.nodes[node_index];
        (
            n.split_dim,
            n.split,
            n.children,
            n.is_leaf(),
            n.less,
            n.greater,
            n.start_idx,
            n.end_idx,
        )
    }

    /// Original point indices stored in a (leaf) node.
    fn node_indices(&self, node_index: usize) -> Vec<usize> {
        let n = &self.inner.nodes[node_index];
        self.inner.raw_indices[n.start_idx..n.end_idx].to_vec()
    }

    /// Flat row-major data of the points in a (leaf) node.
    fn node_data(&self, node_index: usize) -> Vec<f64> {
        let n = &self.inner.nodes[node_index];
        let m = self.inner.m;
        let mut out = Vec::with_capacity((n.end_idx - n.start_idx) * m);
        for i in n.start_idx..n.end_idx {
            let idx = self.inner.raw_indices[i];
            out.extend_from_slice(&self.inner.raw_data[idx * m..(idx + 1) * m]);
        }
        out
    }

    /// Indices in `self` within radius `r` of each query point in `x`.
    fn query_ball_point(
        &self,
        x: &Bound<'_, PyAny>,
        r: f64,
        p: f64,
        eps: f64,
    ) -> PyResult<Vec<Vec<usize>>> {
        let x_flat = extract_f64_flat(x)?;
        let m = self.inner.m;
        let nq = x_flat.len() / m.max(1);
        let upper_bound = to_pow(r, p);
        let ef = epsfac(eps, p);
        let mut results = Vec::with_capacity(nq);
        for q in 0..nq {
            let xq = &x_flat[q * m..(q + 1) * m];
            let mut x_min = vec![0.0f64; m];
            let mut x_max = vec![0.0f64; m];
            x_min.copy_from_slice(xq);
            x_max.copy_from_slice(xq);
            let mut out = Vec::new();
            self.inner
                .query_ball_point_dfs(xq, &x_min, &x_max, 0, upper_bound, ef, p, &mut out);
            results.push(out);
        }
        Ok(results)
    }

    /// For each point in `self`, indices in `other` within radius `r`.
    fn query_ball_tree(
        &self,
        other: &KdTreeCore,
        r: f64,
        p: f64,
        eps: f64,
    ) -> PyResult<Vec<Vec<usize>>> {
        let upper_bound = to_pow(r, p);
        let ef = epsfac(eps, p);
        let mut results: Vec<Vec<usize>> = vec![vec![]; self.inner.n];
        self.inner
            .query_ball_tree_dfs(&other.inner, 0, 0, upper_bound, ef, p, &mut results);
        for list in results.iter_mut() {
            list.sort_unstable();
        }
        Ok(results)
    }

    /// All pairs `(i, j)` with `i < j` within radius `r`.
    fn query_pairs(&self, r: f64, p: f64, eps: f64) -> PyResult<Vec<(usize, usize)>> {
        let upper_bound = to_pow(r, p);
        let ef = epsfac(eps, p);
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        self.inner
            .query_pairs_dfs(0, 0, upper_bound, ef, p, &mut pairs);
        Ok(pairs)
    }

    /// `(i, j, true_distance)` for all pairs within `max_distance`.
    fn sparse_distance_matrix(
        &self,
        other: &KdTreeCore,
        max_distance: f64,
        p: f64,
    ) -> PyResult<Vec<(usize, usize, f64)>> {
        let upper_bound = to_pow(max_distance, p);
        let mut out: Vec<(usize, usize, f64)> = Vec::new();
        self.inner
            .sparse_distance_matrix_dfs(&other.inner, 0, 0, upper_bound, p, &mut out);
        Ok(out)
    }

    /// Count neighbor pairs binned by radius. `real_r` is sorted ascending in `d**p`
    /// units; returns a results array of the same length (accumulated from zero).
    fn count_neighbors(
        &self,
        other: &KdTreeCore,
        real_r: Vec<f64>,
        p: f64,
        weights_self: Option<Vec<f64>>,
        weights_other: Option<Vec<f64>>,
        cumulative: bool,
    ) -> PyResult<Vec<f64>> {
        let sw: Option<Vec<f64>> = weights_self;
        let ow: Option<Vec<f64>> = weights_other;
        let snw = sw.as_ref().map(|w| self.inner.node_weight_sum(w));
        let onw = ow.as_ref().map(|w| other.inner.node_weight_sum(w));
        let mut results = vec![0.0f64; real_r.len()];
        self.inner.count_neighbors_dfs(
            &other.inner,
            0,
            0,
            &real_r,
            p,
            sw.as_deref(),
            ow.as_deref(),
            snw.as_deref(),
            onw.as_deref(),
            cumulative,
            &mut results,
        );
        Ok(results)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<KdTreeCore>()?;
    Ok(())
}
