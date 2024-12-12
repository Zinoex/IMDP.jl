abstract type AbstractCuWorkspace end

###################
# Dense workspace #
###################
struct CuDenseWorkspace <: AbstractCuWorkspace
    max_actions::Int32
end

IntervalMDP.construct_workspace(
    prob::IntervalProbabilities{R, VR, MR},
    max_actions = 1,
) where {R, VR, MR <: AbstractGPUMatrix{R}} = CuDenseWorkspace(max_actions)

struct CuOrthogonalDenseWorkspace <: AbstractCuWorkspace
    max_actions::Int32
end

IntervalMDP.construct_workspace(
    prob::OrthogonalIntervalProbabilities{N, <:IntervalProbabilities{R, VR, MR}},
    max_actions = 1,
) where {N, R, VR, MR <: AbstractGPUMatrix{R}, } = CuOrthogonalDenseWorkspace(max_actions)

####################
# Sparse workspace #
####################
struct CuSparseWorkspace <: AbstractCuWorkspace
    max_nonzeros::Int32
    max_actions::Int32
end

function CuSparseWorkspace(p::AbstractCuSparseMatrix, max_actions)
    max_nonzeros = maximum(nnz, eachcol(p))
    return CuSparseWorkspace(max_nonzeros, max_actions)
end

IntervalMDP.construct_workspace(
    prob::IntervalProbabilities{R, VR, MR},
    max_actions = 1,
) where {R, VR, MR <: AbstractCuSparseMatrix{R}} = CuSparseWorkspace(gap(prob), max_actions)
