from sympy import symbols, sqrt, lambdify, Function, solve, Derivative
from scipy.integrate import solve_ivp

def model_to_H0rc(model):
    return float(model.strip('N').replace('p','.'))

def get_param(param, cosmo):
    return float(cosmo.split(param)[-1].split('_')[0])

def GetLinPred(cosmo_str, model, z_eval, z_ini=499, z_fin=0, H0rc=None):
    # Symbolic variables
    x, y, mu= symbols(r' x, y, \mu')
    a= symbols('a', positive=True)
    D= Function('D')

    # Cosmology and costants
    Om0 = get_param( 'Om', cosmo_str)
    Ol0 = 1- Om0
    H0_hinvMpc= 1/2997.92458
    
    # Internal quantities
    a_ini = 1/(1+z_ini)
    a_fin = 1/(1+z_fin)
    a_eval = 1/(1+z_eval)
    
    # Time dependent functions
    H = H0_hinvMpc*sqrt(Om0*a**(-3) + Ol0)
    H_conf = H*a
    Om = Om0*a**(-3)
    Ol = Ol0
    
    # Choose mu
    if model=='GR':
        mu=1
    elif model=='nDGP':
        # Define MG parameters and differential eq for D
        rc = H0rc/H0_hinvMpc
        beta = 1+2*rc*H*(1+H_conf*H.diff(a)/(3*H**2))
        mu = 1 + 1/(3*beta)
    
    # Set up differential equation for D
    diff_eq = a*H_conf*(a*H_conf*D(a).diff(a)).diff(a) + a*H_conf**2*D(a).diff(a) -3/2*mu*Om*H0_hinvMpc**2*a**(2)*D(a)
    diff_eq = diff_eq.expand()

    # Split 2nd order differential equation in a system of first order differential equation
    x_sym_eq = diff_eq.subs(D(a).diff(a),x).subs(D(a),y)
    x_eq= lambdify((a,x,y), solve(x_sym_eq, Derivative(x,a))[0])
    y_eq = lambdify((a,x,y), x)

    def dum_fun(t,vec):
        '''Dummy function to adapt the input of solve_ivp'''
        return (x_eq(t,vec[0],vec[1]),y_eq(t,vec[0],vec[1]))

    # Compute the solution of the differential equation
    sol = solve_ivp(dum_fun,t_span=(a_ini,a_fin),y0=(1,a_ini), t_eval=a_eval, rtol=1e-9)

    return sol['y'][1]