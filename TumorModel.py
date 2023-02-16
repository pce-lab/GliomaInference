import time
start_time = time.time()
import math
from fenics import *
import dolfin as dl
import ufl
import numpy as np
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.pyplot as plt
import argparse
import itertools as iter
import sys
import os

sep = "\n"+"#"*80+"\n"
from hippylib import *
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tumor Model')
    parser.add_argument('--s',
                        default=0.5,
                        type=float,
                        help="gpCN hyper-parameter")
    args = parser.parse_args()
    try:
        dl.set_log_active(False)
    except:
        pass
    sep = "\n"+"#"*80+"\n"
    ndim = 2

s = float(args.s)

sigma = 0.1
out_dir = "output/"
t_init = 2.
t_final = 9.
tlist = [2.0,4.0,5.0,6.0,9.0]


class TumorEquationVarf:
    def __init__(self, dt, dx):
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        self.dx = dx


    @property
    def dt(self):
        return self._dt

    def __call__(self,u,u_old, m, p, t):
        m1,m2 = dl.split(m)
        return (u - u_old)*p*self.dt_inv*self.dx \
               + ufl.exp(m1)*ufl.inner(ufl.grad(u), ufl.grad(p))*self.dx \
               - ufl.exp(m2)*u*(dl.Constant(1.) - u)*p*self.dx \

# read meshing
# mesh = dl.Mesh(mesh_dir+name+"_"+str(c)+".xml")
mesh = dl.Mesh("../meshing/W05_0.xml")

# set up Function Spaces
P1 = dl.FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=1)
Vh2 = dl.FunctionSpace(mesh, P1)
Vh1 = dl.FunctionSpace(mesh, P1*P1)

Vh = [Vh2, Vh1, Vh2]
ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
print (sep, "Set up the mesh and finite element spaces", sep)
print ("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs) )
# IC and BC
f = dl.Constant(0.0)
bc = []; bc0 = []
# prior mean

# initialize PDE
print(sep,"Running Combination: k = {0}, rhoGM = {1}, sigma = {2} for rat {3}".format( k, rhoGM, sigma, str(rat)), sep)
#####

gamma = 1.0; delta = 1.0

mu = dl.Function(Vh[PARAMETER]); mu.assign(dl.Constant([-2,-3]))

prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, mean = mu.vector(), robin_bc=True)

# load data
data = {};
with dl.XDMFFile("../data/W05.xdmf") as fid:
    for t in tlist:
        foo = dl.Function(Vh[STATE])
        fid.read(foo, "W05")
        data[str(t)] = foo

u0 = data{"2"}

pde_varf = TumorEquationVarf(dt, ufl.dx)
pde = TimeDependentPDEVariationalProblem(Vh, pde_varf, bc, bc0, u0, t_init, t_final, is_fwd_linear=False)

#misfit
misfits = []; l = len(misfittlist)
for t in pde.times:
    misfit_t = ContinuousStateObservation(Vh[STATE], ufl.dx, bc0)
    if t in tlist[0:len(tlist)-1]:
        misfit_t.d.axpy(1., data[str(t)].vector())
    else:
        misfit_t.W.zero()
    misfit_t.noise_variance = sigma**2
    misfits.append(misfit_t)

misfit = MisfitTD(misfits, pde.times)

model = Model(pde, prior, misfit)

print( sep, "FE Difference Check", sep)

Vhm1 = dl.FunctionSpace(mesh, P1)
Vhm2 = dl.FunctionSpace(mesh, P1)
m10_expr = dl.Expression("sin(x[0])", degree=1 )
m20_expr = dl.Expression("sin(x[0])", degree=1 )
m10 = dl.interpolate( m10_expr , Vhm1)
m20 = dl.interpolate( m20_expr , Vhm2)
m0 = dl.Function(Vh[PARAMETER])
dl.assign(m0.sub(0), m10)
dl.assign(m0.sub(1), m20)
modelVerify(model, mu.vector(), is_quadratic = False, misfit_only=True, verbose = 1 )
plt.savefig('modelVerify.png')


print( sep, "Find the MAP point", sep)
m = model.prior.mean.copy()
parameters = ReducedSpaceNewtonCG_ParameterList()
parameters["rel_tolerance"] = 1e-4
parameters["abs_tolerance"] = 1e-4
parameters["max_iter"]      = 100
parameters["globalization"] = "LS"
parameters["GN_iter"] = 100
parameters.showMe()

solver = ReducedSpaceNewtonCG(model, parameters)

x = solver.solve([None, m, None])
t_map = time.time() - start_time

print ("Termination reason: ", solver.termination_reasons[solver.reason])
print ("Final gradient norm: ", solver.final_grad_norm)
print ("Final cost: ", solver.final_cost)
print ("Final simulation time: ", t_map)
convergence = solver.converged

print( "\nConverged in ", solver.it, " iterations.")

pde.exportState(x[STATE], out_dir+"states_"+str(rat)+"_"+str(args.label)+".xdmf")
map = vector2Function(x[PARAMETER], Vh[PARAMETER], name="MAP")
with dl.XDMFFile(out_dir+"MAP.xdmf") as fid:
    fid.write_checkpoint(map,"map")

print (sep, "Compute the low rank Gaussian Approximation of the posterior", sep)

model.setPointForHessianEvaluations(x, gauss_newton_approx = True)
Hmisfit = ReducedHessian(model, misfit_only=True)
k = 50
p = 20

print ("Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )

Omega = MultiVector(x[PARAMETER], k+p)
parRandom.normal(1., Omega)

d, eU = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
posterior = GaussianLRPosterior(prior, d, eU)
posterior.mean = x[PARAMETER]
eU.export(Vh[PARAMETER], out_dir+"evect_"+str(label)+".xdmf", varname = "gen_evects", normalize = True)
np.savetxt(out_dir+"eigevalues_"+str(label)+".dat", d)

print( sep, "Generate Samples from LA", sep )

nsamples = 1000
noise = dl.Vector()
posterior.init_vector(noise,"noise")
s_prior = dl.Function(Vh[PARAMETER], name="sample_prior")
s_post = dl.Function(Vh[PARAMETER], name="LA_post")
s_state = dl.Function(Vh[STATE], name="state")

fid_LA = dl.XDMFFile(out_dir+"s_LA.xdmf")
fid_state = dl.XDMFFile(out_dir+"s_state.xdmf")
nta_slist = []; dice_slist = []
for i in tqdm.tqdm(range(nsamples)):
    parRandom.normal(1., noise)
    posterior.sample(noise, s_prior.vector(), s_post.vector())

    fid_LA.write(s_post, "s_post", i)

    u = pde.generate_state(); x = [u, s_post.vector(), None]
    pde.solveFwd(x[STATE], x); x[STATE].retrieve(s_state.vector(),t_final)

    fid_state.write(s_state, "s_state", i)

print(sep,'Total simulation time = ',time.time() - start_time, sep)

print(sep,'Simulation time = ',time.time() - start_time,sep)
