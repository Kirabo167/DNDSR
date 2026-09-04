"""Independent symbolic derivation checks for variable-density ACM.

Author: Runzhi Ma
Date: 2026-09-03
No project Python/C++ bindings are imported. Run with a Python providing SymPy.
The script prints exact identities and fails on any symbolic discrepancy.
"""
import sympy as sp


def verify():
    """Verify flux, Gamma, eigensystem, pseudo-time product rule and Roe secant.

    Parameters: none. Returns: none. Raises AssertionError on a failed identity.
    """
    rho, beta2 = sp.symbols("rho beta2", positive=True)
    mx, my, mz, p, alpha, lam = sp.symbols("mx my mz p alpha lambda", real=True)
    state = sp.Matrix([rho, mx, my, mz, p])
    velocity = sp.Matrix([mx, my, mz]) / rho
    q = velocity[0]
    flux = sp.Matrix([mx, mx*mx/rho+p, mx*my/rho, mx*mz/rho, mx/rho])
    A = flux.jacobian(state)
    gamma = sp.eye(5)
    gamma[0,4] = (alpha+1)*rho/beta2
    for i in range(3):
        gamma[i+1,4] = (2*alpha+1)*state[i+1]/beta2
    gamma[4,4] = 1/beta2
    inverse = sp.eye(5)
    inverse[0,4] = -(alpha+1)*rho
    for i in range(3):
        inverse[i+1,4] = -(2*alpha+1)*state[i+1]
    inverse[4,4] = beta2
    assert sp.simplify(inverse*gamma) == sp.eye(5)
    J = sp.eye(5)
    J[1:4,0] = velocity
    J[1:4,1:4] = rho*sp.eye(3)
    B = sp.simplify(J.inv()*inverse*A*J)
    expected = sp.Matrix([
        [q,-alpha*rho,0,0,0],
        [0,(1-alpha)*q,0,0,1/rho],
        [0,-alpha*velocity[1],q,0,0],
        [0,-alpha*velocity[2],0,q,0],
        [0,beta2,0,0,0],
    ])
    assert sp.simplify(B-expected) == sp.zeros(5)
    determinant = sp.factor((lam*sp.eye(5)-B).det())
    assert sp.simplify(determinant-(lam-q)**3*(lam**2-(1-alpha)*q*lam-beta2/rho)) == 0
    print("PASS: analytic A = dF/dU; Gamma inverse; primitive B; characteristic polynomial")
    lm, lp = sp.symbols("lambda_minus lambda_plus", real=True)
    def acoustic(root):
        """Return one pressure-normalized primitive acoustic vector for symbolic checks."""
        return sp.Matrix([-alpha*rho*root/(beta2*(root-q)), root/beta2,
                          -alpha*velocity[1]*root/(beta2*(root-q)),
                          -alpha*velocity[2]*root/(beta2*(root-q)), 1])
    for root in (lm, lp):
        relation = {beta2: rho*root*(root-(1-alpha)*q)}
        assert all(sp.factor(v.subs(relation)) == 0 for v in (B-root*sp.eye(5))*acoustic(root))
    contacts = [sp.Matrix([1,0,0,0,0]), sp.Matrix([0,0,1,0,0]),
                sp.Matrix([0,0,0,1,0])]
    assert all(sp.simplify((B-q*sp.eye(5))*r) == sp.zeros(5,1) for r in contacts)
    print("PASS: general-alpha primitive acoustic/contact/shear right eigenvectors")

    old = sp.Matrix(sp.symbols("rho0 mx0 my0 mz0 p0"))
    tau = sp.symbols("tau", positive=True)
    product = gamma * (state-old) / tau
    exact_product_jacobian = product.jacobian(state)
    correction = sp.zeros(5)
    correction[0,0] = (alpha+1)*(p-old[4])/beta2
    for i in range(1,4):
        correction[i,i] = (2*alpha+1)*(p-old[4])/beta2
    assert sp.simplify(exact_product_jacobian-(gamma+correction)/tau) == sp.zeros(5)
    print("PASS: state-dependent Gamma pseudo-time product-rule Jacobian")
    sl,sr = sp.symbols("sL sR",positive=True)
    ul,vl,wl,pl,ur,vr,wr,pr = sp.symbols("uL vL wL pL uR vR wR pR",real=True)
    left = sp.Matrix([sl**2,sl**2*ul,sl**2*vl,sl**2*wl,pl])
    right = sp.Matrix([sr**2,sr**2*ur,sr**2*vr,sr**2*wr,pr])
    mean_u = (sl*sp.Matrix([ul,vl,wl])+sr*sp.Matrix([ur,vr,wr]))/(sl+sr)
    mean = sp.Matrix([sl*sr, *(sl*sr*mean_u), (pl+pr)/2])
    sub = lambda vector: dict(zip(state, vector))
    secant = A.subs(sub(mean), simultaneous=True)*(right-left) - (
        flux.subs(sub(right), simultaneous=True)-flux.subs(sub(left), simultaneous=True))
    assert all(sp.factor(v)==0 for v in secant)
    print("PASS: square-root-density Roe secant for all five physical flux components")
    print("Primitive preconditioned Jacobian:")
    sp.pprint(B)
    print("Characteristic polynomial:", determinant)

    # Two direct counterexamples to formulas found in the supplied reports.
    wrong_flux = flux.copy(); wrong_flux[4] = mx/rho**2
    assert sp.simplify(wrong_flux[4]-flux[4]) != 0
    wrong_contact = sp.Matrix([1,0,0,0,0])
    conservative_B = sp.simplify(inverse*A)
    assert sp.simplify((conservative_B-q*sp.eye(5))*wrong_contact) != sp.zeros(5,1)
    print("PASS: report counterexamples (u/rho flux and conservative [1,0,0,0,0] contact)")


if __name__ == "__main__":
    verify()
