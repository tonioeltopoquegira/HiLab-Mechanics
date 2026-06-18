#!/usr/bin/env python3
"""
fracture_dolfinx.py — DOLFINx phase-field fracture driver for topopt designs.

Wraps the fracture_solver/PF_BrittleFracture submodule's formulation and runs
it on a mesh produced by grid_to_mesh.py.
"""

import sys, os, time, argparse
import numpy as np


from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
import ufl
import basix.ufl
import dolfinx
from dolfinx import fem, mesh as dmesh, plot as dplot
import dolfinx.fem.petsc
from dolfinx.io import XDMFFile
from dolfinx.cpp.log import LogLevel, log

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Imports from Umar's submodule's
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "fracture_solver"))
from pf_core import (SNESSolver, petsc_options_SNES_u, petsc_options_SNES_z,
                     norm_L2, derive_material, build_residuals)



def build_mesh_from_npz(npz_path, comm=MPI.COMM_WORLD):
    """Build a 2D triangular DOLFINx mesh from grid_to_mesh's mesh.npz."""
    data = np.load(npz_path)
    pts = data["points"].astype(np.float64)         
    tris = data["triangles"].astype(np.int64)       
    c_el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    
    domain = dmesh.create_mesh(comm, tris, ufl.Mesh(c_el), pts)
    return domain, pts



#  Driver

def run(npz_path, out_dir="dolfinx_output",
        E=1.0, nu=0.3, Gc=1.0e-3, sts=0.05, scs=0.5, eps=0.02,
        max_disp=0.05, n_steps=200, snapshot_every=10, tol=1e-7):

    comm = MPI.COMM_WORLD; rank = comm.rank
    os.makedirs(out_dir, exist_ok=True)
    domain, pts = build_mesh_from_npz(npz_path, comm)
    fdim = domain.topology.dim - 1                   

    # Material + physics (imported from pf_core)
    p = derive_material(domain, E, nu, Gc, sts, scs, eps)

    V = fem.functionspace(domain, ("Lagrange", 1, (2,))) # displacement vector field (2 dimensional), linearly interpolated inside mesh triangles
    Y = fem.functionspace(domain, ("Lagrange", 1)) # phase field z, a scalar for each mesh triangle, linearly interpolated inside mesh triangles

    u = fem.Function(V, name="displacement"); v = ufl.TestFunction(V); du = ufl.TrialFunction(V) # actual displacement field u, test function v (for weak form), trial function du
    z = fem.Function(Y, name="phasefield");   y = ufl.TestFunction(Y); dz = ufl.TrialFunction(Y) # phase field z, test function y (for weak form), trial function dz
    z.x.array[:] = 1.0  # 1 = intact
    z_trial = fem.Function(Y)
    z_lb = fem.Function(Y); z_lb.x.array[:] = 0.0 # lower bounds on z
    z_ub = fem.Function(Y); z_ub.x.array[:] = 1.0 # upper bounds on z

    # Boundary conditions from geometry (whole bottom / whole top) --> divide them between ranks
    x = domain.geometry.x
    ymin, ymax = x[:, 1].min(), x[:, 1].max()
    ymin = comm.allreduce(ymin, op=MPI.MIN); ymax = comm.allreduce(ymax, op=MPI.MAX)
    tolc = 1e-6 + 0.25*float(np.min(np.abs(np.diff(np.unique(pts[:, 1])))) or 1e-3)

    bot_facets = dmesh.locate_entities_boundary(domain, fdim,
                    lambda p: p[1] <= ymin + tolc)
    top_facets = dmesh.locate_entities_boundary(domain, fdim,
                    lambda p: p[1] >= ymax - tolc)

    disp_ = fem.Constant(domain, ScalarType(0.0)) # the ramped load value (mutable)
    bx0 = fem.locate_dofs_topological(V.sub(0), fdim, bot_facets)  # returns bottom ux
    by0 = fem.locate_dofs_topological(V.sub(1), fdim, bot_facets)  # returns bottom uy
    tyL = fem.locate_dofs_topological(V.sub(1), fdim, top_facets)  # returns top uy (load)
    bcs_u = [
        fem.dirichletbc(ScalarType(0.0), bx0, V.sub(0)),  # bottom fully clamp ux
        fem.dirichletbc(ScalarType(0.0), by0, V.sub(1)),  # bottom fully clamp uy
        fem.dirichletbc(disp_,           tyL, V.sub(1)),  # top is prescribed by displacement disp_ (downward)
    ]
    bcs_z = [] # only bounded by z_lb, z_ub (irreversibility) — no Dirichlet BCs                                               

    if rank == 0:
        print(f"mesh: {domain.topology.index_map(2).size_global} cells | "
              f"y∈[{ymin:.4f},{ymax:.4f}] | support facets={bot_facets.size} "
              f"load facets={top_facets.size} | lch={p['lch']:.4f} eps={eps}")

    # This is code straioght from Umar's submodule, which builds the residuals and Jacobians for the staggered solve
    R_u, J_u, R_z, J_z = build_residuals(u, z, v, y, du, dz, p)
    problem_u = SNESSolver(R_u, u, bcs=bcs_u, J_form=J_u,
                           petsc_options=petsc_options_SNES_u, prefix="u")
    problem_z = SNESSolver(R_z, z, bcs=bcs_z, J_form=J_z, bounds=(z_lb, z_ub),
                           petsc_options=petsc_options_SNES_z, prefix="z")

    # Some matlab utilities to plot the mesh and snapshots of the phase field
    topo, _, geom = dplot.vtk_mesh(Y)          # dof coords + VTK connectivity
    tri_conn = topo.reshape(-1, 4)[:, 1:4]      # triangle type → [3, n0, n1, n2]
    triang = mtri.Triangulation(geom[:, 0], geom[:, 1], tri_conn)
    snaps = []

    def save_snapshot(step, t, F, zmin):
        # plot the native phase field z (1 = intact, 0 = fully cracked)
        fig, ax = plt.subplots(figsize=(11, 4))
        tpc = ax.tripcolor(triang, z.x.array, cmap="RdYlGn", vmin=0, vmax=1,
                           shading="gouraud")
        plt.colorbar(tpc, ax=ax, fraction=0.025, label="phase field z  (1=intact, 0=crack)")
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"step {step}  δ={t*max_disp:.4f}  F={F:.4e}  "
                     f"z_min={zmin:.3f}", fontsize=10)
        plt.tight_layout()
        fn = os.path.join(out_dir, f"phasefield_step{step:04d}.png")
        fig.savefig(fn, dpi=130); plt.close(fig); snaps.append(fn)

    # Load stepping (staggered)
    xdmf = XDMFFile(domain.comm, os.path.join(out_dir, "fracture.xdmf"), "w")
    xdmf.write_mesh(domain)
    fd_path = os.path.join(out_dir, "force_disp.txt")
    if rank == 0:
        open(fd_path, "w").close()
        for stale in __import__("glob").glob(os.path.join(out_dir, "phasefield_step*.png")):
            os.remove(stale)

    t0 = time.time()
    for step in range(1, n_steps+1): # load steps (load incremenets)
        t = step/n_steps # normalized pseudo-time [0, 1]
        disp_.value = -t*max_disp      # downward displacement that grows linearly to max_disp

        # irreversibility: lock already-broken region (z can't heal)
        broken = z.x.array <= 0.05 # find nodes already (nearly) cracked
        z_ub.x.array[broken] = z.x.array[broken] + 0.002 # cap their upper bound at ~current
        z_ub.x.scatter_forward()

        zres, it = 1e9, 0 # big initial residual, iter counter
        while it < 100 and zres > tol: # alternate minimization until z stops changing
            problem_u.solve(); u.x.scatter_forward() # z fixed → solve elasticity for u
            z_trial.x.array[:] = z.x.array
            problem_z.solve(); z.x.scatter_forward() # u fixed → solve phase field for z
            zres = norm_L2(comm, z_trial - z); it += 1 # check how much z changed: if big, keep alternating

        b_e = fem.petsc.assemble_vector(fem.form(-R_u)) # assemble the elasticity residual
        b_e.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE) # sum contributions across ranks
        b_e.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD) # redistribute
        with b_e.localForm() as bl:
            F_top = comm.allreduce(np.sum(bl.array[tyL]), op=MPI.SUM) # sum over top dofs

        # If zmin is close to 0, we consider the material fully fractured and stop the simulation 
        zmin = comm.allreduce(z.x.array.min(), op=MPI.MIN)

        # logging
        if rank == 0:
            with open(fd_path, "a") as f:
                f.write(f"{t:.5f} {t*max_disp:.6f} {F_top:.8e} {zmin:.5f}\n")
            print(f"step {step:4d}/{n_steps}  δ={t*max_disp:.5f}  F={F_top:+.6e}  "
                  f"z_min={zmin:.4f}  stag_it={it}  ({time.time()-t0:.1f}s)", flush=True)

        broke = zmin < 0.02
        if step % snapshot_every == 0 or step == n_steps or broke:
            xdmf.write_function(u, t); xdmf.write_function(z, t)
            if rank == 0: save_snapshot(step, t, F_top, zmin)
        if broke:
            if rank == 0: print("  ✓ fully fractured — stopping."); break

    xdmf.close()

    # damage snapshots
    if rank == 0 and snaps:
        n = len(snaps); cols = min(n, 4); rows = (n + cols - 1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 2.1*rows))
        for ax, fn in zip(np.array(axes).ravel(), snaps):
            ax.imshow(plt.imread(fn)); ax.axis("off")
        for ax in np.array(axes).ravel()[n:]: ax.axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "phasefield_overview.png"), dpi=110)
        plt.close(fig)
    if rank == 0:
        print(f"Done in {time.time()-t0:.1f}s → {out_dir}/  "
              f"(fracture.xdmf, force_disp.txt, phasefield_step*.png, phasefield_overview.png)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="?", default="mesh_output/mesh.npz",
                   help="mesh.npz produced by grid_to_mesh.py")
    p.add_argument("--out-dir", default="dolfinx_output")

    # Can pick Young modulus
    p.add_argument("--E",   type=float, default=1.0)
    # Can pick Poisson ratio
    p.add_argument("--nu",  type=float, default=0.3)
    # Can pick the critical energy release rate (fracture toughness)
    p.add_argument("--Gc",  type=float, default=1.0e-3)
    # Can pick the tensile strength
    p.add_argument("--sts", type=float, default=0.05)
    # Can pick the compressive strength
    p.add_argument("--scs", type=float, default=0.5)
    # Can pick the regularization length for the phase field
    p.add_argument("--eps", type=float, default=0.02)
    # Can pick the maximum displacement to apply at the top
    p.add_argument("--max-disp", type=float, default=0.05)
    # Can pick the number of load steps to apply
    p.add_argument("--steps", type=int, default=200)
    # Can pick how often to save snapshots (XDMF + PNG)
    p.add_argument("--snapshot-every", type=int, default=10)

    a = p.parse_args()
    run(a.npz, a.out_dir, E=a.E, nu=a.nu, Gc=a.Gc, sts=a.sts, scs=a.scs, eps=a.eps,
        max_disp=a.max_disp, n_steps=a.steps, snapshot_every=a.snapshot_every)
