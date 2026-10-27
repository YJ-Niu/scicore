"""
A top-level linear programming interface.

.. versionadded:: 0.15.0

Functions
---------
.. autosummary::
   :toctree: generated/

    linprog
    linprog_verbose_callback
    linprog_terse_callback

"""

import rsnumpy as np

from ._optimize import OptimizeResult, OptimizeWarning
from warnings import warn
from ._linprog_highs import _linprog_highs
from ._linprog_ip import _linprog_ip
from ._linprog_simplex import _linprog_simplex
from ._linprog_rs import _linprog_rs
from ._linprog_doc import (_linprog_highs_doc, _linprog_ip_doc,  # noqa: F401
                           _linprog_rs_doc, _linprog_simplex_doc,
                           _linprog_highs_ipm_doc, _linprog_highs_ds_doc)
from ._linprog_util import (
    _parse_linprog, _presolve, _get_Abc, _LPProblem, _autoscale,
    _postsolve, _check_result, _display_summary)
from copy import deepcopy

__all__ = ['linprog', 'linprog_verbose_callback', 'linprog_terse_callback']

__docformat__ = "restructuredtext en"

LINPROG_METHODS = [
    'simplex', 'revised simplex', 'interior-point', 'highs', 'highs-ds', 'highs-ipm'
]


def _linprog_fallback(lp, solver_options):
    """Solve the original problem with a pure-Python two-phase simplex.

    Fallback for the HiGHS methods when the HiGHS backend is unavailable
    (the C++ extension was replaced by a stub). Deliberately works on
    plain Python floats and lists, avoiding parts of the rsnumpy array
    API that are not yet reliable.

    Returns the same result dict as ``_linprog_highs``.
    """
    maxiter = solver_options.get('maxiter', 1000)
    maxiter = int(maxiter) if maxiter is not None else 1000
    tol = solver_options.get('tol', 1e-9)
    tol = float(tol) if tol is not None else 1e-9

    c = [float(x) for x in np.asarray(lp.c).tolist()]
    A_eq = np.asarray(lp.A_eq).tolist() if lp.A_eq is not None else []
    b_eq = [float(x) for x in np.asarray(lp.b_eq).tolist()] \
        if lp.b_eq is not None else []
    A_ub = np.asarray(lp.A_ub).tolist() if lp.A_ub is not None else []
    b_ub = [float(x) for x in np.asarray(lp.b_ub).tolist()] \
        if lp.b_ub is not None else []
    bounds = [[float(x) for x in row]
              for row in np.asarray(lp.bounds).tolist()]
    integrality = (np.asarray(lp.integrality).tolist()
                   if lp.integrality is not None else [0] * len(c))
    if any(int(v) != 0 for v in integrality):
        raise NotImplementedError(
            "integer constraints require the HiGHS backend, which is "
            "not yet ported")

    n = len(c)
    neg_inf = float('-inf')
    pos_inf = float('inf')

    # --- make every variable non-negative ------------------------------
    #   free: x = p - q     shift/box: x = lb + y     flip: x = ub - y
    # 'box' (finite lb and ub) additionally gets a row y + z = ub - lb.
    trans = []
    names = []
    cstd = []
    j_of = {}
    c0 = 0.0
    for i in range(n):
        lb, ub = bounds[i]
        j_of[i] = len(cstd)
        if lb == neg_inf:
            if ub == pos_inf:
                trans.append(('free', i))
                names += [('free_pos', i), ('free_neg', i)]
                cstd += [c[i], -c[i]]
            else:
                trans.append(('flip', i, ub))
                names.append(('flip', i, ub))
                cstd.append(-c[i])
                c0 += c[i] * ub
        elif ub == pos_inf:
            trans.append(('shift', i, lb))
            names.append(('shift', i, lb))
            cstd.append(c[i])
            c0 += c[i] * lb
        else:
            trans.append(('box', i, lb))
            names.append(('box', i, lb))
            cstd.append(c[i])
            c0 += c[i] * lb

    def unmap(y):
        x = [0.0] * n
        for j, name in enumerate(names):
            kind = name[0]
            if kind == 'free_pos':
                x[name[1]] = y[j]
            elif kind == 'free_neg':
                x[name[1]] -= y[j]
            elif kind in ('shift', 'box'):
                x[name[1]] = name[2] + y[j]
            elif kind == 'flip':
                x[name[1]] = name[2] - y[j]
        for i in range(n):
            lb, ub = bounds[i]
            if lb != neg_inf and lb - 1e-9 < x[i] < lb:
                x[i] = lb
            if ub != pos_inf and ub < x[i] < ub + 1e-9:
                x[i] = ub
        return x

    n_vars = len(names)
    ub_rows = []  # (column of y, rhs) for the finite upper bounds
    for j, name in enumerate(names):
        if name[0] == 'box':
            i, lb = name[1], name[2]
            ub_rows.append((j, float(bounds[i][1]) - lb))
    m_ub = len(ub_rows)
    m_ineq = len(A_ub)
    # zero-cost slack variables: upper-bound rows first, then A_ub rows
    cstd += [0.0] * (m_ub + m_ineq)
    n_cols = n_vars + m_ub + m_ineq

    def transform_row(row):
        out = [0.0] * n_vars
        rhs = 0.0
        for i, a in enumerate(row):
            a = float(a)
            if a == 0.0:
                continue
            t = trans[i]
            if t[0] == 'free':
                out[j_of[i]] += a
                out[j_of[i] + 1] -= a
            elif t[0] == 'flip':
                out[j_of[i]] -= a
                rhs += a * t[2]
            else:  # shift / box
                out[j_of[i]] += a
                rhs += a * t[2]
        return out, rhs

    rows = []
    rhs_vec = []
    for row, b in zip(A_eq, b_eq):
        base, b_ = transform_row(row)
        rows.append(base + [0.0] * (m_ub + m_ineq))
        rhs_vec.append(float(b) - b_)
    for s, (row, b) in enumerate(zip(A_ub, b_ub)):
        base, b_ = transform_row(row)
        r = base + [0.0] * (m_ub + m_ineq)
        r[n_vars + m_ub + s] = 1.0
        rows.append(r)
        rhs_vec.append(float(b) - b_)
    for t, (j, b) in enumerate(ub_rows):
        r = [0.0] * n_cols
        r[j] = 1.0
        r[n_vars + t] = 1.0
        rows.append(r)
        rhs_vec.append(b)
    m = len(rows)

    ok_message = ('Optimization terminated successfully. '
                  '(simplex fallback, HiGHS not yet ported)')

    def finish(status, message, x_orig, fun, nit):
        if x_orig is None:
            return {'x': None, 'fun': None, 'status': status,
                    'slack': np.array([]), 'con': np.array([]),
                    'nit': nit, 'message': message}
        slack = [float(b) - sum(float(a) * x for a, x in zip(row, x_orig))
                 for row, b in zip(A_ub, b_ub)]
        con = [float(b) - sum(float(a) * x for a, x in zip(row, x_orig))
               for row, b in zip(A_eq, b_eq)]
        return {'x': np.array(x_orig), 'fun': float(fun),
                'status': status, 'slack': np.array(slack),
                'con': np.array(con), 'nit': nit, 'message': message}

    if m == 0:
        x0 = unmap([0.0] * n_cols)
        fun0 = sum(ci * xi for ci, xi in zip(c, x0))
        if any(cv < -tol for cv in cstd):
            return finish(3, 'Problem appears to be unbounded.',
                          None, None, 0)
        return finish(0, ok_message, x0, fun0, 0)

    # --- two-phase simplex on the standard-form problem ----------------
    # dictionary: x_{B_r} = rhs[r] - sum_j table[r][j] x_j
    n_art = m
    var_n = n_cols + n_art
    table = []
    rhs = []
    basis = []
    obj_val = 0.0
    obj = {}
    for r in range(m):
        row = rows[r]
        b = rhs_vec[r]
        if b >= 0.0:
            d_row = {j: v for j, v in enumerate(row) if v != 0.0}
            rhs.append(b)
            obj_val += b
            for j, d in d_row.items():
                obj[j] = obj.get(j, 0.0) - d
        else:
            # flip the row so the artificial starts at -b > 0; the
            # objective then contains -a_r for this row
            d_row = {j: -v for j, v in enumerate(row) if v != 0.0}
            rhs.append(-b)
            obj_val -= (-b)
            for j, d in d_row.items():
                obj[j] = obj.get(j, 0.0) + d
        table.append(d_row)
        basis.append(n_cols + r)

    def pivot(obj_val, enter, leave):
        r = basis.index(leave)
        a = table[r][enter]
        new_row = {}
        for j in range(var_n):
            if j != enter:
                d = table[r].get(j, 0.0)
                if d != 0.0:
                    new_row[j] = d / a
        new_row[leave] = 1.0 / a
        rhs[r] /= a
        f = obj.pop(enter, 0.0)
        for rr in range(len(table)):
            if rr == r:
                continue
            c_ = table[rr].get(enter, 0.0)
            if c_ == 0.0:
                continue
            rhs[rr] -= c_ * rhs[r]
            table[rr][leave] = table[rr].get(leave, 0.0) + c_ / a
            for j in list(table[rr]):
                if j != leave:
                    table[rr][j] += c_ * new_row.get(j, 0.0)
                    if table[rr][j] == 0.0:
                        del table[rr][j]
        obj[leave] = obj.get(leave, 0.0) - f / a
        for j in list(obj):
            if j != leave:
                obj[j] -= f * new_row.get(j, 0.0)
                if obj[j] == 0.0:
                    del obj[j]
        table[r] = new_row
        basis[r] = enter
        return obj_val + f * rhs[r]

    def simplex(obj_val, scan_n, iters):
        """Bland's-rule primal simplex; returns (status, nit, obj_val)."""
        basic_set = set(basis)
        nit = 0
        for _ in range(iters):
            enter = None
            for j in range(scan_n):
                if j not in basic_set and obj.get(j, 0.0) < -tol:
                    enter = j
                    break
            if enter is None:
                return 0, nit, obj_val
            best = None
            for r in range(len(rhs)):
                d = table[r].get(enter, 0.0)
                if d > tol:
                    ratio = rhs[r] / d
                    if (best is None or ratio < best[0] - 1e-12 or
                            (abs(ratio - best[0]) <= 1e-12 and
                             basis[r] < basis[best[1]])):
                        best = (ratio, r)
            if best is None:
                return 3, nit, obj_val
            leave = basis[best[1]]
            obj_val = pivot(obj_val, enter, leave)
            basic_set.remove(leave)
            basic_set.add(enter)
            nit += 1
        return 1, nit, obj_val

    # phase one: find a basic feasible solution (min sum of artificials)
    status, nit, obj_val = simplex(obj_val, var_n, maxiter)
    if status == 1:
        return finish(1, 'Iteration limit reached.', None, None, nit)
    if status == 3:  # cannot happen: the phase-one objective is bounded
        return finish(4, 'Serious numerical difficulties encountered.',  # below
                      None, None, nit)
    if obj_val > 100 * tol:
        return finish(2, 'Problem appears to be infeasible.', None, None, nit)

    # basic artificials must leave the basis before phase two
    r = 0
    while r < len(basis):
        if basis[r] >= n_cols:
            cand = None
            for j in sorted(table[r]):
                if j < n_cols and abs(table[r][j]) > tol:
                    cand = j
                    break
            if cand is not None:
                obj_val = pivot(obj_val, cand, basis[r])
            else:  # redundant row
                del table[r]
                del rhs[r]
                del basis[r]
                continue
        r += 1

    # phase two: minimize c0 + cstd . y
    basic_set = set(basis)
    obj = {}
    for j in range(n_cols):
        if j not in basic_set and cstd[j] != 0.0:
            obj[j] = cstd[j]
    obj_val = c0
    for r in range(len(basis)):
        b = basis[r]
        cb = cstd[b]
        if cb == 0.0:
            continue
        obj_val += cb * rhs[r]
        for j, d in table[r].items():
            obj[j] = obj.get(j, 0.0) - cb * d
            if obj[j] == 0.0:
                del obj[j]

    status, nit2, obj_val = simplex(obj_val, n_cols, max(0, maxiter - nit))
    nit += nit2
    if status == 1:
        return finish(1, 'Iteration limit reached.', None, None, nit)
    if status == 3:
        return finish(3, 'Problem appears to be unbounded.', None, None, nit)

    y = [0.0] * n_cols
    for r, b in enumerate(basis):
        if rhs[r] > 0.0:
            y[b] = rhs[r]
        elif rhs[r] > -1e-12:
            y[b] = 0.0
    x_orig = unmap(y)
    fun = sum(ci * xi for ci, xi in zip(c, x_orig))
    return finish(0, ok_message, x_orig, fun, nit)


def linprog_verbose_callback(res):
    """
    A sample callback function demonstrating the linprog callback interface.
    This callback produces detailed output to sys.stdout before each iteration
    and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    res : A `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            The independent variable vector which optimizes the linear
            programming problem.
        fun : float
            Value of the objective function.
        success : bool
            True if the algorithm succeeded in finding an optimal solution.
        slack : 1-D array
            The values of the slack variables. Each slack variable corresponds
            to an inequality constraint. If the slack is zero, then the
            corresponding constraint is active.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints, that is,
            ``b - A_eq @ x``
        phase : int
            The phase of the optimization being executed. In phase 1 a basic
            feasible solution is sought and the T has an additional row
            representing an alternate objective function.
        status : int
            An integer representing the exit status of the optimization:

            ``0`` : Optimization terminated successfully

            ``1`` : Iteration limit reached

            ``2`` : Problem appears to be infeasible

            ``3`` : Problem appears to be unbounded

            ``4`` : Serious numerical difficulties encountered

        nit : int
            The number of iterations performed.
        message : str
            A string descriptor of the exit status of the optimization.
    """
    x = res['x']
    fun = res['fun']
    phase = res['phase']
    status = res['status']
    nit = res['nit']
    message = res['message']
    complete = res['complete']

    saved_printoptions = np.get_printoptions()
    np.set_printoptions(linewidth=500,
                        formatter={'float': lambda x: f"{x: 12.4f}"})
    if status:
        print('--------- Simplex Early Exit -------\n')
        print(f'The simplex method exited early with status {status:d}')
        print(message)
    elif complete:
        print('--------- Simplex Complete --------\n')
        print(f'Iterations required: {nit}')
    else:
        print(f'--------- Iteration {nit:d}  ---------\n')

    if nit > 0:
        if phase == 1:
            print('Current Pseudo-Objective Value:')
        else:
            print('Current Objective Value:')
        print('f = ', fun)
        print()
        print('Current Solution Vector:')
        print('x = ', x)
        print()

    np.set_printoptions(**saved_printoptions)


def linprog_terse_callback(res):
    """
    A sample callback function demonstrating the linprog callback interface.
    This callback produces brief output to sys.stdout before each iteration
    and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    res : A `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            The independent variable vector which optimizes the linear
            programming problem.
        fun : float
            Value of the objective function.
        success : bool
            True if the algorithm succeeded in finding an optimal solution.
        slack : 1-D array
            The values of the slack variables. Each slack variable corresponds
            to an inequality constraint. If the slack is zero, then the
            corresponding constraint is active.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints, that is,
            ``b - A_eq @ x``.
        phase : int
            The phase of the optimization being executed. In phase 1 a basic
            feasible solution is sought and the T has an additional row
            representing an alternate objective function.
        status : int
            An integer representing the exit status of the optimization:

            ``0`` : Optimization terminated successfully

            ``1`` : Iteration limit reached

            ``2`` : Problem appears to be infeasible

            ``3`` : Problem appears to be unbounded

            ``4`` : Serious numerical difficulties encountered

        nit : int
            The number of iterations performed.
        message : str
            A string descriptor of the exit status of the optimization.
    """
    nit = res['nit']
    x = res['x']

    if nit == 0:
        print("Iter:   X:")
    print(f"{nit: <5d}   ", end="")
    print(x)


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
            bounds=(0, None), method='highs', callback=None,
            options=None, x0=None, integrality=None):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints.

    Linear programming solves problems of the following form:

    .. math::

        \min_x \ & c^T x \\
        \mbox{such that} \ & A_{ub} x \leq b_{ub},\\
        & A_{eq} x = b_{eq},\\
        & l \leq x \leq u ,

    where :math:`x` is a vector of decision variables; :math:`c`,
    :math:`b_{ub}`, :math:`b_{eq}`, :math:`l`, and :math:`u` are vectors; and
    :math:`A_{ub}` and :math:`A_{eq}` are matrices.

    Alternatively, that's:

    - minimize ::

        c @ x

    - such that ::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub

    Note that by default ``lb = 0`` and ``ub = None``. Other bounds can be
    specified with ``bounds``.

    Parameters
    ----------
    c : 1-D array
        The coefficients of the linear objective function to be minimized.
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    A_eq : 2-D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1-D array, optional
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    bounds : sequence, optional
        A sequence of ``(min, max)`` pairs for each element in ``x``, defining
        the minimum and maximum values of that decision variable.
        If a single tuple ``(min, max)`` is provided, then ``min`` and ``max``
        will serve as bounds for all decision variables.
        Use ``None`` to indicate that there is no bound. For instance, the
        default bound ``(0, None)`` means that all decision variables are
        non-negative, and the pair ``(None, None)`` means no bounds at all,
        i.e. all variables are allowed to be any real.
    method : str, optional
        The algorithm used to solve the standard form problem.
        The following are supported.

        - :ref:`'highs' <optimize.linprog-highs>` (default)
        - :ref:`'highs-ds' <optimize.linprog-highs-ds>`
        - :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`
        - :ref:`'interior-point' <optimize.linprog-interior-point>` (legacy)
        - :ref:`'revised simplex' <optimize.linprog-revised_simplex>` (legacy)
        - :ref:`'simplex' <optimize.linprog-simplex>` (legacy)

        The legacy methods are deprecated and will be removed in a future
        version of SciPy.
    callback : callable, optional
        If a callback function is provided, it will be called at least once per
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            The current solution vector.
        fun : float
            The current value of the objective function ``c @ x``.
        success : bool
            ``True`` when the algorithm has completed successfully.
        slack : 1-D array
            The (nominally positive) values of the slack,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        phase : int
            The phase of the algorithm being executed.
        status : int
            An integer representing the status of the algorithm.

            ``0`` : Optimization proceeding nominally.

            ``1`` : Iteration limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : Numerical difficulties encountered.

        nit : int
            The current iteration number.
        message : str
            A string descriptor of the algorithm status.

        Callback functions are not currently supported by the HiGHS methods.

    options : dict, optional
        A dictionary of solver options. All methods accept the following
        options:

        maxiter : int
            Maximum number of iterations to perform.
            Default: see method-specific documentation.
        disp : bool
            Set to ``True`` to print convergence messages.
            Default: ``False``.
        presolve : bool
            Set to ``False`` to disable automatic presolve.
            Default: ``True``.

        All methods except the HiGHS solvers also accept:

        tol : float
            A tolerance which determines when a residual is "close enough" to
            zero to be considered exactly zero.
        autoscale : bool
            Set to ``True`` to automatically perform equilibration.
            Consider using this option if the numerical values in the
            constraints are separated by several orders of magnitude.
            Default: ``False``.
        rr : bool
            Set to ``False`` to disable automatic redundancy removal.
            Default: ``True``.
        rr_method : str
            Method used to identify and remove redundant rows from the
            equality constraint matrix after presolve. For problems with
            dense input, the available methods for redundancy removal are:

            ``SVD``:
                Repeatedly performs singular value decomposition on
                the matrix, detecting redundant rows based on nonzeros
                in the left singular vectors that correspond with
                zero singular values. May be fast when the matrix is
                nearly full rank.
            ``pivot``:
                Uses the algorithm presented in [5]_ to identify
                redundant rows.
            ``ID``:
                Uses a randomized interpolative decomposition.
                Identifies columns of the matrix transpose not used in
                a full-rank interpolative decomposition of the matrix.
            ``None``:
                Uses ``svd`` if the matrix is nearly full rank, that is,
                the difference between the matrix rank and the number
                of rows is less than five. If not, uses ``pivot``. The
                behavior of this default is subject to change without
                prior notice.

            Default: None.
            For problems with sparse input, this option is ignored, and the
            pivot-based algorithm presented in [5]_ is used.

        For method-specific options, see
        :func:`show_options('linprog') <show_options>`.

    x0 : 1-D array, optional
        Guess values of the decision variables, which will be refined by
        the optimization algorithm. This argument is currently used only by the
        :ref:`'revised simplex' <optimize.linprog-revised_simplex>` method,
        and can only be used if `x0` represents a basic feasible solution.

    integrality : 1-D array or int, optional
        Indicates the type of integrality constraint on each decision variable.

        ``0`` : Continuous variable; no integrality constraint.

        ``1`` : Integer variable; decision variable must be an integer
        within `bounds`.

        ``2`` : Semi-continuous variable; decision variable must be within
        `bounds` or ``0``.

        ``3`` : Semi-integer variable; decision variable must be an integer
        within `bounds` or ``0``.

        By default, all variables are continuous.

        For mixed integrality constraints, supply an array of shape ``c.shape``.
        To infer a constraint on each decision variable from shorter inputs,
        the argument will be broadcast to ``c.shape`` using `rsnumpy.broadcast_to`.

        This argument is currently used only by the
        :ref:`'highs' <optimize.linprog-highs>` method and is ignored otherwise.

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields
        below. Note that the return types of the fields may depend on whether
        the optimization was successful, therefore it is recommended to check
        `OptimizeResult.status` before relying on the other fields:

        x : 1-D array
            The values of the decision variables that minimizes the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        slack : 1-D array
            The (nominally positive) values of the slack variables,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        success : bool
            ``True`` when the algorithm succeeds in finding an optimal
            solution.
        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimization terminated successfully.

            ``1`` : Iteration limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : Numerical difficulties encountered.

        nit : int
            The total number of iterations performed in all phases.
        message : str
            A string descriptor of the exit status of the algorithm.

    See Also
    --------
    show_options : Additional options accepted by the solvers.

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter.

    :ref:`'highs-ds' <optimize.linprog-highs-ds>`, and
    :ref:`'highs-ipm' <optimize.linprog-highs-ipm>` are interfaces to the
    HiGHS simplex and interior-point method solvers [13]_, respectively.
    :ref:`'highs' <optimize.linprog-highs>` (default) chooses between
    the two automatically. These are the fastest linear
    programming solvers in SciPy, especially for large, sparse problems;
    which of these two is faster is problem-dependent.
    The other solvers are legacy methods and will be removed when `callback` is
    supported by the HiGHS methods.

    Method :ref:`'highs-ds' <optimize.linprog-highs-ds>`, is a wrapper of the C++ high
    performance dual revised simplex implementation (HSOL) [13]_, [14]_.
    Method :ref:`'highs-ipm' <optimize.linprog-highs-ipm>` is a wrapper of a C++
    implementation of an **i**\ nterior-\ **p**\ oint **m**\ ethod [13]_; it
    features a crossover routine, so it is as accurate as a simplex solver.
    Method :ref:`'highs' <optimize.linprog-highs>` chooses between the two
    automatically.
    For new code involving `linprog`, we recommend explicitly choosing one of
    these three method values.

    .. versionadded:: 1.6.0

    Method :ref:`'interior-point' <optimize.linprog-interior-point>`
    uses the primal-dual path following algorithm
    as outlined in [4]_. This algorithm supports sparse constraint matrices and
    is typically faster than the simplex methods, especially for large, sparse
    problems. Note, however, that the solution returned may be slightly less
    accurate than those of the simplex methods and will not, in general,
    correspond with a vertex of the polytope defined by the constraints.

    .. versionadded:: 1.0.0

    Method :ref:`'revised simplex' <optimize.linprog-revised_simplex>`
    uses the revised simplex method as described in
    [9]_, except that a factorization [11]_ of the basis matrix, rather than
    its inverse, is efficiently maintained and used to solve the linear systems
    at each iteration of the algorithm.

    .. versionadded:: 1.3.0

    Method :ref:`'simplex' <optimize.linprog-simplex>` uses a traditional,
    full-tableau implementation of
    Dantzig's simplex algorithm [1]_, [2]_ (*not* the
    Nelder-Mead simplex). This algorithm is included for backwards
    compatibility and educational purposes.

    .. versionadded:: 0.15.0

    Before applying :ref:`'interior-point' <optimize.linprog-interior-point>`,
    :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, or
    :ref:`'simplex' <optimize.linprog-simplex>`,
    a presolve procedure based on [8]_ attempts
    to identify trivial infeasibilities, trivial unboundedness, and potential
    problem simplifications. Specifically, it checks for:

    - rows of zeros in ``A_eq`` or ``A_ub``, representing trivial constraints;
    - columns of zeros in ``A_eq`` `and` ``A_ub``, representing unconstrained
      variables;
    - column singletons in ``A_eq``, representing fixed variables; and
    - column singletons in ``A_ub``, representing simple bounds.

    If presolve reveals that the problem is unbounded (e.g. an unconstrained
    and unbounded variable has negative cost) or infeasible (e.g., a row of
    zeros in ``A_eq`` corresponds with a nonzero in ``b_eq``), the solver
    terminates with the appropriate status code. Note that presolve terminates
    as soon as any sign of unboundedness is detected; consequently, a problem
    may be reported as unbounded when in reality the problem is infeasible
    (but infeasibility has not been detected yet). Therefore, if it is
    important to know whether the problem is actually infeasible, solve the
    problem again with option ``presolve=False``.

    If neither infeasibility nor unboundedness are detected in a single pass
    of the presolve, bounds are tightened where possible and fixed
    variables are removed from the problem. Then, linearly dependent rows
    of the ``A_eq`` matrix are removed, (unless they represent an
    infeasibility) to avoid numerical difficulties in the primary solve
    routine. Note that rows that are nearly linearly dependent (within a
    prescribed tolerance) may also be removed, which can change the optimal
    solution in rare cases. If this is a concern, eliminate redundancy from
    your problem formulation and run with option ``rr=False`` or
    ``presolve=False``.

    Several potential improvements can be made here: additional presolve
    checks outlined in [8]_ should be implemented, the presolve routine should
    be run multiple times (until no further simplifications can be made), and
    more of the efficiency improvements from [5]_ should be implemented in the
    redundancy removal routines.

    After presolve, the problem is transformed to standard form by converting
    the (tightened) simple bounds to upper bound constraints, introducing
    non-negative slack variables for inequality constraints, and expressing
    unbounded variables as the difference between two non-negative variables.
    Optionally, the problem is automatically scaled via equilibration [12]_.
    The selected algorithm solves the standard form problem, and a
    postprocessing routine converts the result to a solution to the original
    problem.

    References
    ----------
    .. [1] Dantzig, George B., Linear programming and extensions. Rand
           Corporation Research Study Princeton Univ. Press, Princeton, NJ,
           1963
    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
           Mathematical Programming", McGraw-Hill, Chapter 4.
    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
           Mathematics of Operations Research (2), 1977: pp. 103-107.
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [5] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    .. [7] Fourer, Robert. "Solving Linear Programs by Interior-Point Methods."
           Unpublished Course Notes, August 26, 2005. Available 2/25/2017 at
           http://www.4er.org/CourseNotes/Book%20B/B-III.pdf
    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
           programming." Mathematical Programming 71.2 (1995): 221-245.
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [10] Andersen, Erling D., et al. Implementation of interior point
            methods for large scale linear programming. HEC/Universite de
            Geneve, 1996.
    .. [11] Bartels, Richard H. "A stabilization of the simplex method."
            Journal in  Numerische Mathematik 16.5 (1971): 414-434.
    .. [12] Tomlin, J. A. "On scaling linear programming problems."
            Mathematical Programming Study 4 (1975): 146-166.
    .. [13] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
            "HiGHS - high performance software for linear optimization."
            https://highs.dev/
    .. [14] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
            simplex method." Mathematical Programming Computation, 10 (1),
            119-142, 2018. :doi:`10.1007/s12532-017-0130-5`.

    Examples
    --------
    Consider the following problem:

    .. math::

        \min_{x_0, x_1} \ -x_0 + 4x_1 & \\
        \mbox{such that} \ -3x_0 + x_1 & \leq 6,\\
        -x_0 - 2x_1 & \geq -4,\\
        x_1 & \geq -3.

    The problem is not presented in the form accepted by `linprog`. This is
    easily remedied by converting the "greater than" inequality
    constraint to a "less than" inequality constraint by
    multiplying both sides by a factor of :math:`-1`. Note also that the last
    constraint is really the simple bound :math:`-3 \leq x_1 \leq \infty`.
    Finally, since there are no bounds on :math:`x_0`, we must explicitly
    specify the bounds :math:`-\infty \leq x_0 \leq \infty`, as the
    default is for variables to be non-negative. After collecting coefficients
    into arrays and tuples, the input for this problem is:

    >>> from scicore.optimize import linprog
    >>> c = [-1, 4]
    >>> A = [[-3, 1], [1, 2]]
    >>> b = [6, 4]
    >>> x0_bounds = (None, None)
    >>> x1_bounds = (-3, None)
    >>> res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
    >>> res.fun
    -22.0
    >>> res.x
    array([10., -3.])
    >>> res.message
    'Optimization terminated successfully. (HiGHS Status 7: Optimal)'

    The marginals (AKA dual values / shadow prices / Lagrange multipliers)
    and residuals (slacks) are also available.

    >>> res.ineqlin
      residual: [ 3.900e+01  0.000e+00]
     marginals: [-0.000e+00 -1.000e+00]

    For example, because the marginal associated with the second inequality
    constraint is -1, we expect the optimal value of the objective function
    to decrease by ``eps`` if we add a small amount ``eps`` to the right hand
    side of the second inequality constraint:

    >>> eps = 0.05
    >>> b[1] += eps
    >>> linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds]).fun
    -22.05

    Also, because the residual on the first inequality constraint is 39, we
    can decrease the right hand side of the first constraint by 39 without
    affecting the optimal solution.

    >>> b = [6, 4]  # reset to original values
    >>> b[0] -= 39
    >>> linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds]).fun
    -22.0

    """

    meth = method.lower()
    methods = {"highs", "highs-ds", "highs-ipm",
               "simplex", "revised simplex", "interior-point"}

    if meth not in methods:
        raise ValueError(f"Unknown solver '{method}'")

    if x0 is not None and meth != "revised simplex":
        warning_message = "x0 is used only when method is 'revised simplex'. "
        warn(warning_message, OptimizeWarning, stacklevel=2)

    if integrality is None:
        integrality = np.zeros(len(c), dtype=int)
    else:
        integrality = np.asarray(integrality)

    if np.any(integrality) and not meth == "highs":
        integrality = None
        warning_message = ("Only `method='highs'` supports integer "
                           "constraints. Ignoring `integrality`.")
        warn(warning_message, OptimizeWarning, stacklevel=2)
    elif np.any(integrality):
        integrality = np.broadcast_to(integrality, np.shape(c))
    else:
        integrality = None

    lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality)
    lp, solver_options = _parse_linprog(lp, options, meth)
    tol = solver_options.get('tol', 1e-9)

    # Give unmodified problem to HiGHS
    if meth.startswith('highs'):
        if callback is not None:
            raise NotImplementedError("HiGHS solvers do not support the "
                                      "callback interface.")
        highs_solvers = {'highs-ipm': 'ipm', 'highs-ds': 'simplex',
                         'highs': None}

        try:
            sol = _linprog_highs(lp, solver=highs_solvers[meth],
                                 **solver_options)
        except (NotImplementedError, AttributeError):
            # The HiGHS backend is not yet ported (the C++ extension was
            # replaced by a stub); solve with a pure-Python two-phase
            # simplex on the original problem instead.
            sol = _linprog_fallback(lp, solver_options)
        sol['status'], sol['message'] = (
            _check_result(sol['x'], sol['fun'], sol['status'], sol['slack'],
                          sol['con'], lp.bounds, tol, sol['message'],
                          integrality))
        sol['success'] = sol['status'] == 0
        return OptimizeResult(sol)

    warn(f"`method='{meth}'` is deprecated and will be removed in a future version "
         "of SciPy. Please use one of the HiGHS solvers (e.g. `method='highs'`) in "
         "new code.", DeprecationWarning, stacklevel=2)

    iteration = 0
    complete = False  # will become True if solved in presolve
    undo = []

    # Keep the original arrays to calculate slack/residuals for original
    # problem.
    lp_o = deepcopy(lp)

    # Solve trivial problem, eliminate variables, tighten bounds, etc.
    rr_method = solver_options.pop('rr_method', None)  # need to pop these;
    rr = solver_options.pop('rr', True)  # they're not passed to methods
    c0 = 0  # we might get a constant term in the objective
    if solver_options.pop('presolve', True):
        (lp, c0, x, undo, complete, status, message) = _presolve(lp, rr,
                                                                 rr_method,
                                                                 tol)

    C, b_scale = 1, 1  # for trivial unscaling if autoscale is not used
    postsolve_args = (lp_o._replace(bounds=lp.bounds), undo, C, b_scale)

    if not complete:
        A, b, c, c0, x0 = _get_Abc(lp, c0)
        if solver_options.pop('autoscale', False):
            A, b, c, x0, C, b_scale = _autoscale(A, b, c, x0)
            postsolve_args = postsolve_args[:-2] + (C, b_scale)

        if meth == 'simplex':
            x, status, message, iteration = _linprog_simplex(
                c, c0=c0, A=A, b=b, callback=callback,
                postsolve_args=postsolve_args, **solver_options)
        elif meth == 'interior-point':
            x, status, message, iteration = _linprog_ip(
                c, c0=c0, A=A, b=b, callback=callback,
                postsolve_args=postsolve_args, **solver_options)
        elif meth == 'revised simplex':
            x, status, message, iteration = _linprog_rs(
                c, c0=c0, A=A, b=b, x0=x0, callback=callback,
                postsolve_args=postsolve_args, **solver_options)

    # Eliminate artificial variables, re-introduce presolved variables, etc.
    disp = solver_options.get('disp', False)

    x, fun, slack, con = _postsolve(x, postsolve_args, complete)

    status, message = _check_result(x, fun, status, slack, con, lp_o.bounds,
                                    tol, message, integrality)

    if disp:
        _display_summary(message, status, fun, iteration)

    sol = {
        'x': x,
        'fun': fun,
        'slack': slack,
        'con': con,
        'status': status,
        'message': message,
        'nit': iteration,
        'success': status == 0}

    return OptimizeResult(sol)
