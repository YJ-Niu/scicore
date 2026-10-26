//! 2-D computational geometry replacing the deleted C ``_qhull`` extension.
//!
//! * convex hull: Graham scan
//! * Delaunay triangulation: incremental Bowyer–Watson
//! * Voronoi diagram: dual of the Delaunay triangulation
//!
//! Python wrappers: ``scicore/spatial/_qhull.py``.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

type Pt = (f64, f64);

fn points_from_bytes(bytes: &[u8], n: usize, m: usize) -> PyResult<Vec<Pt>> {
    if m != 2 {
        return Err(PyValueError::new_err(
            "geometry: only 2-D point sets are supported",
        ));
    }
    if bytes.len() != n * 2 * std::mem::size_of::<f64>() {
        return Err(PyValueError::new_err("geometry: buffer length mismatch"));
    }
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 16;
        let x = f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        let y = f64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());
        if !x.is_finite() || !y.is_finite() {
            return Err(PyValueError::new_err("geometry: non-finite coordinates"));
        }
        pts.push((x, y));
    }
    Ok(pts)
}

fn check_distinct(pts: &[Pt]) -> PyResult<()> {
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            if pts[i] == pts[j] {
                return Err(PyValueError::new_err(
                    "geometry: duplicate points are not supported",
                ));
            }
        }
    }
    Ok(())
}

fn cross(o: Pt, a: Pt, b: Pt) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

fn dist2(a: Pt, b: Pt) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

// ---------------------------------------------------------------------------
// Convex hull (Graham scan)
// ---------------------------------------------------------------------------

/// Graham scan; returns the hull indices in counter-clockwise order and the
/// hull area.  Raises an error when the input is degenerate.
fn convex_hull_2d(pts: &[Pt]) -> PyResult<(Vec<usize>, f64)> {
    let n = pts.len();
    if n < 3 {
        return Err(PyValueError::new_err(
            "geometry: at least 3 points are needed for a 2-D hull",
        ));
    }
    // Pivot: lowest y, then lowest x.
    let mut pivot = 0usize;
    for i in 1..n {
        if pts[i].1 < pts[pivot].1
            || (pts[i].1 == pts[pivot].1 && pts[i].0 < pts[pivot].0)
        {
            pivot = i;
        }
    }
    let mut rest: Vec<usize> = (0..n).filter(|&i| i != pivot).collect();
    rest.sort_by(|&a, &b| {
        let c = cross(pts[pivot], pts[a], pts[b]);
        if c > 0.0 {
            std::cmp::Ordering::Less
        } else if c < 0.0 {
            std::cmp::Ordering::Greater
        } else {
            // Collinear: the closer point first.
            dist2(pts[pivot], pts[a])
                .partial_cmp(&dist2(pts[pivot], pts[b]))
                .unwrap()
        }
    });
    let mut hull: Vec<usize> = Vec::with_capacity(n);
    for idx in std::iter::once(pivot).chain(rest.iter().copied()) {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            if cross(pts[a], pts[b], pts[idx]) <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(idx);
    }
    if hull.len() < 3 {
        return Err(PyValueError::new_err(
            "geometry: input is degenerate (collinear)",
        ));
    }
    let mut area2 = 0.0f64;
    for i in 0..hull.len() {
        let (x1, y1) = pts[hull[i]];
        let (x2, y2) = pts[hull[(i + 1) % hull.len()]];
        area2 += x1 * y2 - x2 * y1;
    }
    Ok((hull, area2.abs() * 0.5))
}

// ---------------------------------------------------------------------------
// Delaunay triangulation (incremental Bowyer-Watson)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Tri {
    a: usize,
    b: usize,
    c: usize,
}

impl Tri {
    fn edges(&self) -> [(usize, usize); 3] {
        [(self.a, self.b), (self.b, self.c), (self.c, self.a)]
    }
}

/// In-circle test for a counter-clockwise triangle: true when ``p`` lies
/// inside the circumcircle.
fn incircle_contains(tri: &Tri, pts: &[Pt], p: Pt) -> bool {
    let (ax, ay) = pts[tri.a];
    let (bx, by) = pts[tri.b];
    let (cx, cy) = pts[tri.c];
    let (px, py) = p;
    let axp = ax - px;
    let ayp = ay - py;
    let bxp = bx - px;
    let byp = by - py;
    let cxp = cx - px;
    let cyp = cy - py;
    let det = (axp * axp + ayp * ayp) * (bxp * cyp - byp * cxp)
        - (bxp * bxp + byp * byp) * (axp * cyp - ayp * cxp)
        + (cxp * cxp + cyp * cyp) * (axp * byp - ayp * bxp);
    det > 0.0
}

fn circumcenter(pts: &[Pt], tri: &Tri) -> Pt {
    let (ax, ay) = pts[tri.a];
    let (bx, by) = pts[tri.b];
    let (cx, cy) = pts[tri.c];
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1.0e-15 {
        return ((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0);
    }
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    (ux, uy)
}

/// Incremental Bowyer-Watson Delaunay triangulation.  Returns the triangles
/// (counter-clockwise, input indices) and their circumcenters.
fn delaunay_2d(pts: &[Pt]) -> (Vec<Tri>, Vec<Pt>) {
    let n = pts.len();
    let (mut minx, mut maxx) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut miny, mut maxy) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in pts {
        minx = minx.min(x);
        maxx = maxx.max(x);
        miny = miny.min(y);
        maxy = maxy.max(y);
    }
    let cx = (minx + maxx) * 0.5;
    let cy = (miny + maxy) * 0.5;
    let d = 4.0 * (maxx - minx).max(maxy - miny).max(1.0);
    // Super-triangle (counter-clockwise), strictly containing all points.
    let s0 = (cx - d, cy - 0.5 * d);
    let s1 = (cx, cy + d);
    let s2 = (cx + d, cy - 0.5 * d);
    let mut all = pts.to_vec();
    all.push(s0);
    all.push(s1);
    all.push(s2);
    let (sn0, sn1, sn2) = (n, n + 1, n + 2);

    let mut tris: Vec<Tri> = vec![Tri { a: sn0, b: sn2, c: sn1 }];
    for p in 0..n {
        let pp: Pt = pts[p];
        let mut bad: Vec<bool> = vec![false; tris.len()];
        let mut bad_list: Vec<usize> = Vec::new();
        for (ti, tri) in tris.iter().enumerate() {
            if incircle_contains(tri, &all, pp) {
                bad[ti] = true;
                bad_list.push(ti);
            }
        }
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
        for &ti in &bad_list {
            for (u, v) in tris[ti].edges() {
                *edge_count.entry((u, v)).or_insert(0) += 1;
            }
        }
        let mut next: Vec<Tri> = Vec::with_capacity(tris.len() + 1);
        for (ti, tri) in tris.iter().enumerate() {
            if !bad[ti] {
                next.push(*tri);
            }
        }
        for (&(u, v), &cnt) in &edge_count {
            if cnt == 1 {
                let t = if cross(all[u], all[v], pp) >= 0.0 {
                    Tri { a: u, b: v, c: p }
                } else {
                    Tri { a: v, b: u, c: p }
                };
                next.push(t);
            }
        }
        tris = next;
    }
    // Drop super-triangle triangles and numerically degenerate ones.
    tris.retain(|t| {
        t.a < n
            && t.b < n
            && t.c < n
            && (cross(all[t.a], all[t.b], all[t.c])).abs() > 1.0e-12 * d * d
    });
    let centers = tris.iter().map(|t| circumcenter(&all, t)).collect();
    (tris, centers)
}

// ---------------------------------------------------------------------------
// Voronoi diagram (dual of the Delaunay triangulation)
// ---------------------------------------------------------------------------

/// Voronoi diagram as the dual of a Delaunay triangulation.
///
/// Returns (all vertices including the synthetic ones at infinity, finite
/// vertices, ridge vertex index pairs, ridge endpoint coordinates, and
/// per-point cell vertex lists); unbounded ridge ends use -1 / -1.0.
fn voronoi_2d(
    pts: &[Pt],
    tris: &[Tri],
    centers: &[Pt],
) -> (
    Vec<Pt>,
    Vec<Pt>,
    Vec<(i64, i64)>,
    Vec<(Pt, Option<Pt>)>,
    Vec<Vec<i64>>,
) {
    let n = pts.len();
    let (mut minx, mut maxx) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut miny, mut maxy) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in pts {
        minx = minx.min(x);
        maxx = maxx.max(x);
        miny = miny.min(y);
        maxy = maxy.max(y);
    }
    let d = (maxx - minx).max(maxy - miny).max(1.0);
    let r_inf = 1.0e4 * d;

    // Finite vertices: deduplicated circumcenters.
    let finite_tol = 1.0e-9 * d;
    let mut finite: Vec<Pt> = Vec::new();
    let mut tri_finite: Vec<usize> = Vec::new();
    for &c in centers {
        if let Some(pos) = finite.iter().position(|&f| {
            (f.0 - c.0).abs() <= finite_tol && (f.1 - c.1).abs() <= finite_tol
        }) {
            tri_finite.push(pos);
        } else {
            tri_finite.push(finite.len());
            finite.push(c);
        }
    }

    // Delaunay edges -> incident triangles.
    let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for (u, v) in t.edges() {
            let key = (u.min(v), u.max(v));
            edge_tris.entry(key).or_default().push(ti);
        }
    }

    // Finite vertices are stored first in ``verts`` so that the finite index
    // doubles as the all-vertex index.
    let mut verts: Vec<Pt> = finite.clone();
    let mut ridge_vertices: Vec<(i64, i64)> = Vec::new();
    let mut ridge_points: Vec<(Pt, Option<Pt>)> = Vec::new();
    let mut cell_verts: Vec<HashSet<usize>> = (0..n).map(|_| HashSet::new()).collect();
    let mut inf_index: HashMap<(i64, i64), usize> = HashMap::new();

    for ((u, v), ti_list) in &edge_tris {
        let ti0 = ti_list[0];
        let cc_idx = tri_finite[ti0];
        let cc = finite[cc_idx];
        // Far end of the ridge: the second circumcenter (interior edge) or a
        // synthetic point at infinity (hull edge).
        let end_idx: usize;
        if ti_list.len() == 2 {
            end_idx = tri_finite[ti_list[1]];
        } else {
            let t = tris[ti0];
            let w = [t.a, t.b, t.c]
                .into_iter()
                .find(|&q| q != *u && q != *v)
                .unwrap();
            let (ux, uy) = pts[*u];
            let (vx, vy) = pts[*v];
            let (wx, wy) = pts[w];
            let ex = vx - ux;
            let ey = vy - uy;
            let len = (ex * ex + ey * ey).sqrt();
            let (mut dx, mut dy) = (-ey / len, ex / len);
            // Point away from the triangle (out of the triangulation).
            if dx * (wx - ux) + dy * (wy - uy) > 0.0 {
                dx = -dx;
                dy = -dy;
            }
            let key = ((dx * 1.0e6).round() as i64, (dy * 1.0e6).round() as i64);
            end_idx = if let Some(&existing) = inf_index.get(&key) {
                existing
            } else {
                inf_index.insert(key, verts.len());
                verts.push((cc.0 + dx * r_inf, cc.1 + dy * r_inf));
                verts.len() - 1
            };
        }
        let bounded = ti_list.len() == 2;
        ridge_vertices.push((
            cc_idx as i64,
            if bounded { end_idx as i64 } else { -1 },
        ));
        ridge_points.push((
            cc,
            if bounded { Some(finite[end_idx]) } else { None },
        ));
        for &ti in ti_list {
            for pv in [tris[ti].a, tris[ti].b, tris[ti].c] {
                cell_verts[pv].insert(cc_idx);
                cell_verts[pv].insert(end_idx);
            }
        }
    }

    // Order the cell vertices by angle around the point.
    let mut regions: Vec<Vec<i64>> = Vec::with_capacity(n);
    for i in 0..n {
        if cell_verts[i].is_empty() {
            regions.push(vec![-1]);
            continue;
        }
        let (px, py) = pts[i];
        let mut idxs: Vec<usize> = cell_verts[i].iter().cloned().collect();
        idxs.sort_by(|&a, &b| {
            let (ax, ay) = verts[a];
            let (bx, by) = verts[b];
            (ay - py)
                .atan2(ax - px)
                .partial_cmp(&(by - py).atan2(bx - px))
                .unwrap()
        });
        regions.push(idxs.into_iter().map(|v| v as i64).collect());
    }

    (verts, finite, ridge_vertices, ridge_points, regions)
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

/// Convex hull of a 2-D point set.
///
/// Returns (hull vertex indices, area, facet equations flattened as
/// [nx, ny, offset, ...]).
#[pyfunction]
fn geometry_convex_hull_2d(bytes: Vec<u8>, n: usize) -> PyResult<(Vec<i64>, f64, Vec<f64>)> {
    let pts = points_from_bytes(&bytes, n, 2)?;
    check_distinct(&pts)?;
    let (hull, area) = convex_hull_2d(&pts)?;
    let mut eq: Vec<f64> = Vec::with_capacity(hull.len() * 3);
    for i in 0..hull.len() {
        let (x1, y1) = pts[hull[i]];
        let (x2, y2) = pts[hull[(i + 1) % hull.len()]];
        let (dx, dy) = (x2 - x1, y2 - y1);
        let len = (dx * dx + dy * dy).sqrt();
        // Outward unit normal for a counter-clockwise polygon.
        let (nx, ny) = (dy / len, -dx / len);
        eq.push(nx);
        eq.push(ny);
        eq.push(-(nx * x1 + ny * y1));
    }
    Ok((hull.iter().map(|&i| i as i64).collect(), area, eq))
}

/// Delaunay triangulation of a 2-D point set.
///
/// Returns (flattened simplices, 3 per triangle; vertex indices; flattened
/// circumcenters, 2 per triangle).
#[pyfunction]
fn geometry_delaunay_2d(
    bytes: Vec<u8>,
    n: usize,
) -> PyResult<(Vec<i64>, Vec<i64>, Vec<f64>)> {
    let pts = points_from_bytes(&bytes, n, 2)?;
    check_distinct(&pts)?;
    let (tris, centers) = delaunay_2d(&pts);
    if tris.is_empty() {
        return Err(PyValueError::new_err(
            "geometry: could not triangulate (degenerate input)",
        ));
    }
    let mut sim: Vec<i64> = Vec::with_capacity(tris.len() * 3);
    let mut seen: HashSet<usize> = HashSet::new();
    for t in &tris {
        sim.extend([t.a as i64, t.b as i64, t.c as i64]);
        seen.extend([t.a, t.b, t.c]);
    }
    let mut verts: Vec<i64> = seen.into_iter().map(|v| v as i64).collect();
    verts.sort_unstable();
    let mut cc: Vec<f64> = Vec::with_capacity(centers.len() * 2);
    for (x, y) in &centers {
        cc.push(*x);
        cc.push(*y);
    }
    Ok((sim, verts, cc))
}

/// Voronoi diagram of a 2-D point set (dual of the Delaunay triangulation).
///
/// Returns (all vertices incl. synthetic ones at infinity; finite vertices;
/// ridge vertex indices; ridge point coordinates; per-point cell vertex
/// lists); unbounded ridge ends are -1 / -1.0.
#[pyfunction]
fn geometry_voronoi_2d(
    bytes: Vec<u8>,
    n: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i64>, Vec<f64>, Vec<Vec<i64>>)> {
    let pts = points_from_bytes(&bytes, n, 2)?;
    check_distinct(&pts)?;
    let (tris, centers) = delaunay_2d(&pts);
    if tris.is_empty() {
        return Err(PyValueError::new_err(
            "geometry: could not triangulate (degenerate input)",
        ));
    }
    let (verts, finite, ridge_vertices, ridge_points, regions) = voronoi_2d(&pts, &tris, &centers);
    let mut vf: Vec<f64> = Vec::with_capacity(verts.len() * 2);
    for (x, y) in &verts {
        vf.push(*x);
        vf.push(*y);
    }
    let mut ff: Vec<f64> = Vec::with_capacity(finite.len() * 2);
    for (x, y) in &finite {
        ff.push(*x);
        ff.push(*y);
    }
    let mut rv: Vec<i64> = Vec::with_capacity(ridge_vertices.len() * 2);
    for (a, b) in &ridge_vertices {
        rv.push(*a);
        rv.push(*b);
    }
    let mut rp: Vec<f64> = Vec::with_capacity(ridge_points.len() * 2);
    for (a, b) in &ridge_points {
        rp.push(a.0);
        rp.push(a.1);
        match b {
            Some((x, y)) => {
                rp.push(*x);
                rp.push(*y);
            }
            None => {
                rp.push(-1.0);
                rp.push(-1.0);
            }
        }
    }
    Ok((vf, ff, rv, rp, regions))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(geometry_convex_hull_2d, module)?)?;
    module.add_function(wrap_pyfunction!(geometry_delaunay_2d, module)?)?;
    module.add_function(wrap_pyfunction!(geometry_voronoi_2d, module)?)?;
    Ok(())
}
