use pyo3::prelude::*;

const INV_PI: f64 = std::f64::consts::FRAC_1_PI;
const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;
const TWO_INV_PI: f64 = std::f64::consts::FRAC_2_PI;

#[pyclass(name = "Nolan")]
struct NolanKernel {
    alpha: f64,
    zeta: f64,
    xi: f64,
    zeta_prefactor: f64,
    alpha_exp: f64,
    alpha_xi: f64,
    zeta_offset: f64,
    two_beta_div_pi: f64,
    pi_div_two_beta: f64,
    x0_div_term: f64,
    c1: f64,
    c2: f64,
    c3: f64,
}

#[pymethods]
impl NolanKernel {
    #[new]
    fn new(alpha: f64, beta: f64, x0: f64) -> Self {
        let zeta = -beta * (HALF_PI * alpha).tan();
        if alpha != 1.0 {
            let xi = (-zeta).atan() / alpha;
            let zeta_prefactor = (zeta.powi(2) + 1.0).powf(-1.0 / (2.0 * (alpha - 1.0)));
            let alpha_exp = alpha / (alpha - 1.0);
            let alpha_xi = (-zeta).atan();
            let zeta_offset = x0 - zeta;
            let (c1, c3) = if alpha < 1.0 {
                (0.5 - xi * INV_PI, INV_PI)
            } else {
                (1.0, -INV_PI)
            };
            let c2 = alpha * INV_PI / (alpha - 1.0).abs() / (x0 - zeta);
            Self {
                alpha,
                zeta,
                xi,
                zeta_prefactor,
                alpha_exp,
                alpha_xi,
                zeta_offset,
                two_beta_div_pi: 0.0,
                pi_div_two_beta: 0.0,
                x0_div_term: 0.0,
                c1,
                c2,
                c3,
            }
        } else {
            let two_beta_div_pi = beta * TWO_INV_PI;
            Self {
                alpha,
                zeta,
                xi: HALF_PI,
                zeta_prefactor: 0.0,
                alpha_exp: 0.0,
                alpha_xi: 0.0,
                zeta_offset: 0.0,
                two_beta_div_pi,
                pi_div_two_beta: HALF_PI / beta,
                x0_div_term: x0 / two_beta_div_pi,
                c1: 0.0,
                c2: 0.5 / beta.abs(),
                c3: INV_PI,
            }
        }
    }

    fn g(&self, theta: f64) -> f64 {
        if self.alpha != 1.0 {
            if theta == -self.xi {
                return if self.alpha < 1.0 { 0.0 } else { f64::INFINITY };
            }
            if theta == HALF_PI {
                return if self.alpha < 1.0 { f64::INFINITY } else { 0.0 };
            }
            let cos_theta = theta.cos();
            self.zeta_prefactor
                * (cos_theta / (self.alpha_xi + self.alpha * theta).sin() * self.zeta_offset)
                    .powf(self.alpha_exp)
                * (self.alpha_xi + (self.alpha - 1.0) * theta).cos()
                / cos_theta
        } else {
            if theta == -self.xi {
                return 0.0;
            }
            if theta == HALF_PI {
                return f64::INFINITY;
            }
            (1.0 + theta * self.two_beta_div_pi)
                * ((self.pi_div_two_beta + theta) * theta.tan() - self.x0_div_term).exp()
                / theta.cos()
        }
    }

    #[getter]
    fn zeta(&self) -> f64 {
        self.zeta
    }

    #[getter]
    fn xi(&self) -> f64 {
        self.xi
    }

    #[getter]
    fn c1(&self) -> f64 {
        self.c1
    }

    #[getter]
    fn c2(&self) -> f64 {
        self.c2
    }

    #[getter]
    fn c3(&self) -> f64 {
        self.c3
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NolanKernel>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{HALF_PI, INV_PI, NolanKernel};

    #[test]
    fn precalculates_alpha_not_one_parameters() {
        let nolan = NolanKernel::new(0.5, 1.0, 1.0);
        assert!((nolan.zeta + 1.0).abs() < 1e-15);
        assert!(nolan.c1.abs() < 1e-15);
        assert!((nolan.c3 - INV_PI).abs() < 1e-15);
        assert_eq!(nolan.g(-nolan.xi), 0.0);
        assert_eq!(nolan.g(HALF_PI), f64::INFINITY);
    }

    #[test]
    fn precalculates_alpha_one_parameters() {
        let nolan = NolanKernel::new(1.0, 0.5, 2.0);
        assert_eq!(nolan.xi, HALF_PI);
        assert_eq!(nolan.c1, 0.0);
        assert!((nolan.c3 - INV_PI).abs() < 1e-15);
        assert_eq!(nolan.g(-HALF_PI), 0.0);
        assert_eq!(nolan.g(HALF_PI), f64::INFINITY);
    }
}
