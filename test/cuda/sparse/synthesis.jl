using Revise, Test
using IntervalMDP, SparseArrays, CUDA

prob1 = IntervalProbabilities(;
    lower = sparse([
        0.0 0.5
        0.1 0.3
        0.2 0.1
    ]),
    upper = sparse([
        0.5 0.7
        0.6 0.5
        0.7 0.3
    ]),
)

prob2 = IntervalProbabilities(;
    lower = sparse([
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ]),
    upper = sparse([
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ]),
)

prob3 = IntervalProbabilities(;
    lower = sparse([
        0.0
        0.0
        1.0
    ][:, :]),
    upper = sparse([
        0.0
        0.0
        1.0
    ][:, :]),
)

transition_probs = [prob1, prob2, prob3]
istates = [Int32(1)]

mdp = IntervalMarkovDecisionProcess(transition_probs, istates)
mdp = IntervalMDP.cu(mdp)

# Finite time reachability
prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
policy, V, k, res = control_synthesis(problem)
cpu_policy = IntervalMDP.cpu(policy)

@test cpu_policy isa TimeVaryingStrategy
@test time_length(cpu_policy) == 10
for k in 1:time_length(cpu_policy)
    @test cpu_policy[k] == [1, 2, 1]
end

# Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
problem = Problem(mdp, spec, policy)
V_mc, k, res = value_iteration(problem)
@test V ≈ V_mc

# Finite time reward
prop = FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10)
prop = IntervalMDP.cu(prop)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)

policy, V, k, res = control_synthesis(problem)
cpu_policy = IntervalMDP.cpu(policy)

@test time_length(cpu_policy) == 10
for k in 1:time_length(cpu_policy)
    @test cpu_policy[k] == [2, 2, 1]
end

# Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
problem = Problem(mdp, spec, policy)
V_mc, k, res = value_iteration(problem)
@test V ≈ V_mc

# Infinite time reachability
prop = InfiniteTimeReachability([3], 1e-6)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
policy, V, k, res = control_synthesis(problem)

@test policy isa StationaryStrategy
@test IntervalMDP.cpu(policy)[1] == [1, 2, 1]

# Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
problem = Problem(mdp, spec, policy)
V_mc, k, res = value_iteration(problem)
@test V ≈ V_mc

# Finite time safety
prop = FiniteTimeSafety([3], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
policy, V, k, res = control_synthesis(problem)
cpu_policy = IntervalMDP.cpu(policy)

@test all(V .>= 0.0)
@test CUDA.@allowscalar(V[3]) ≈ 0.0

@test cpu_policy isa TimeVaryingStrategy
@test time_length(cpu_policy) == 10
for k in 1:(time_length(cpu_policy) - 1)
    @test cpu_policy[k] == [2, 2, 1]
end

# The last time step (aka. the first value iteration step) has a different strategy.
@test cpu_policy[time_length(cpu_policy)] == [2, 1, 1]
