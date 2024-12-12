module IntervalMDPCudaExt

import LLVM
using LLVM.Interop: assume

using CUDA, CUDA.CUSPARSE, Adapt, SparseArrays
using GPUArrays: AbstractGPUArray, AbstractGPUVector, AbstractGPUMatrix

using IntervalMDP, LinearAlgebra

Adapt.@adapt_structure IntervalProbabilities
Adapt.@adapt_structure OrthogonalIntervalProbabilities
Adapt.@adapt_structure MixtureIntervalProbabilities

# Opinionated conversion to GPU with Float64 values and Int32 indices
IntervalMDP.cu(model) = adapt(IntervalMDP.CuModelAdaptor{Float64}, model)
IntervalMDP.cpu(model) = adapt(IntervalMDP.CpuModelAdaptor{Float64}, model)

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mdp::IntervalMarkovDecisionProcess,
)
    return IntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(CuArray{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(CuArray{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CpuModelAdaptor},
    mdp::IntervalMarkovDecisionProcess,
)
    return IntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(Array{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(Array{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mdp::OrthogonalIntervalMarkovDecisionProcess,
)
    return OrthogonalIntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(CuArray{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(CuArray{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CpuModelAdaptor},
    mdp::OrthogonalIntervalMarkovDecisionProcess,
)
    return OrthogonalIntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(Array{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(Array{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mdp::MixtureIntervalMarkovDecisionProcess,
)
    return MixtureIntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(CuArray{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(CuArray{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CpuModelAdaptor},
    mdp::MixtureIntervalMarkovDecisionProcess,
)
    return MixtureIntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(Array{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(Array{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

Adapt.adapt_structure(T::Type{<:IntervalMDP.CuModelAdaptor}, is::AllStates) = is
Adapt.adapt_structure(T::Type{<:IntervalMDP.CpuModelAdaptor}, is::AllStates) = is

Adapt.@adapt_structure StationaryStrategy

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CpuModelAdaptor},
    strategy::TimeVaryingStrategy,
)
    return TimeVaryingStrategy(
        map(s -> adapt(Array{Int32}, s), strategy.strategy),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    strategy::TimeVaryingStrategy,
)
    return TimeVaryingStrategy(
        map(s -> adapt(CuArray{Int32}, s), strategy.strategy),
    )
end


include("cuda/utils.jl")
include("cuda/array.jl")
include("cuda/sorting.jl")
include("cuda/workspace.jl")
include("cuda/strategy.jl")
include("cuda/bellman/dense.jl")
include("cuda/bellman/sparse.jl")
include("cuda/value_iteration.jl")
include("cuda/interval_probabilities.jl")
include("cuda/specification.jl")

end
