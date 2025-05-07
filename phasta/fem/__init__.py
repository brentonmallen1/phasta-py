"""Finite element method module."""

from .shape_function import LagrangeShapeFunction, ShapeFunction
from .metrics import ElementMetrics
from .integration import QuadratureRule, ElementIntegrator
from .assembly import ElementAssembly, GlobalAssembly
from .global_assembly import GlobalAssembly
from .solver import LinearSolver, TimeDependentSolver, NonlinearSolver
from .io import MeshIO, SolutionIO, PHASTAIO
from .parallel import ParallelMesh, ParallelAssembly, ParallelSolver
from .mesh import Mesh
from .solvers import (
    DirectSolver, GMRESSolver,
    ConjugateGradientSolver, BiCGSTABSolver, SolverFactory
)
from .preconditioners import (
    Preconditioner, DiagonalPreconditioner, ILUPreconditioner,
    BlockJacobiPreconditioner, AMGPreconditioner, PreconditionerFactory
)
from .time import (
    TimeIntegrator, ExplicitEuler, ImplicitEuler,
    CrankNicolson, BDF2, TimeDependentProblem
)
from .partitioning import MeshPartitioner, LoadBalancer
from .parallel_io import ParallelIO
from .parallel_solvers import (
    ParallelLinearSolver, ParallelGMRES, ParallelCG, ParallelBiCGSTAB,
    ParallelSolverFactory
)

__all__ = [
    'LagrangeShapeFunction',
    'ShapeFunction',
    'ElementMetrics',
    'QuadratureRule',
    'ElementIntegrator',
    'ElementAssembly',
    'GlobalAssembly',
    'LinearSolver',
    'TimeDependentSolver',
    'NonlinearSolver',
    'MeshIO',
    'SolutionIO',
    'PHASTAIO',
    'ParallelMesh',
    'ParallelAssembly',
    'ParallelSolver',
    'Mesh',
    'DirectSolver',
    'GMRESSolver',
    'ConjugateGradientSolver',
    'BiCGSTABSolver',
    'SolverFactory',
    'Preconditioner',
    'DiagonalPreconditioner',
    'ILUPreconditioner',
    'BlockJacobiPreconditioner',
    'AMGPreconditioner',
    'PreconditionerFactory',
    'TimeIntegrator',
    'ExplicitEuler',
    'ImplicitEuler',
    'CrankNicolson',
    'BDF2',
    'TimeDependentProblem',
    'MeshPartitioner',
    'LoadBalancer',
    'ParallelIO',
    'ParallelLinearSolver',
    'ParallelGMRES',
    'ParallelCG',
    'ParallelBiCGSTAB',
    'ParallelSolverFactory'
]
