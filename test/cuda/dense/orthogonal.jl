using Revise, Test
using IntervalMDP, CUDA
using Random: MersenneTwister

# @testset "bellman 1d" begin
#     prob = IntervalMDP.cu(OrthogonalIntervalProbabilities(
#         (
#             IntervalProbabilities(;
#                 lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
#                 upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
#             ),
#         ),
#         (Int32(2),),
#     ))

#     V = IntervalMDP.cu([1.0, 2.0, 3.0])

#     @testset "maximization" begin
#         Vexpected = [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

#         Vres = bellman(V, prob; upper_bound = true)
#         Vres = IntervalMDP.cpu(Vres)
#         @test Vres ≈ Vexpected

#         Vres = similar(V, 2)
#         bellman!(Vres, V, prob; upper_bound = true)
#         Vres = IntervalMDP.cpu(Vres)
#         @test Vres ≈ Vexpected

#         ws = construct_workspace(prob)
#         strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
#         Vres = similar(V, 2)
#         bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = true)
#         Vres = IntervalMDP.cpu(Vres)
#         @test Vres ≈ Vexpected
#     end

#     @testset "minimization" begin
#         Vexpected = [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

#         Vres = bellman(V, prob; upper_bound = false)
#         Vres = IntervalMDP.cpu(Vres)
#         @test Vres ≈ Vexpected

#         Vres = similar(V, 2)
#         bellman!(Vres, V, prob; upper_bound = false)
#         Vres = IntervalMDP.cpu(Vres)
#         @test Vres ≈ Vexpected

#         ws = construct_workspace(prob)
#         strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
#         Vres = similar(V, 2)
#         bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = false)
#         Vres = IntervalMDP.cpu(Vres)
#         @test Vres ≈ Vexpected
#     end
# end

@testset "bellman 3d" begin
    lower1 = [
        1/15 3/10 1/15 3/10 1/30 1/3 7/30 4/15 1/6 1/5 1/10 1/5 0 7/30 7/30 1/5 2/15 1/6 1/10 1/30 1/10 1/15 1/10 1/15 4/15 4/15 1/3
        1/5 4/15 1/10 1/5 3/10 3/10 1/10 1/15 3/10 3/10 7/30 1/5 1/10 1/5 1/5 1/30 1/5 3/10 1/5 1/5 1/10 1/30 4/15 1/10 1/5 1/6 7/30
        4/15 1/30 1/5 1/5 7/30 4/15 2/15 7/30 1/5 1/3 2/15 1/6 1/6 1/3 4/15 3/10 1/30 3/10 3/10 1/10 1/15 1/30 2/15 1/6 1/5 1/10 4/15
    ]
    lower2 = [
        1/10 1/15 3/10 0 1/6 1/15 1/15 1/6 1/6 1/30 1/10 1/10 1/3 2/15 3/10 4/15 2/15 2/15 1/6 7/30 1/15 2/15 1/10 1/3 7/30 1/30 7/30
        3/10 1/5 3/10 2/15 0 1/30 0 1/15 1/30 7/30 1/30 1/15 7/30 1/15 1/6 1/30 1/10 1/15 3/10 0 3/10 1/6 3/10 1/5 0 7/30 2/15
        3/10 4/15 1/10 3/10 2/15 1/3 3/10 1/10 1/6 3/10 7/30 1/6 1/15 1/15 1/10 1/5 1/5 4/15 1/15 1/3 2/15 1/15 1/5 1/5 1/15 7/30 1/15
    ]
    lower3 = [
        4/15 1/5 3/10 3/10 4/15 7/30 1/5 4/15 7/30 1/6 1/5 0 1/15 1/30 3/10 1/3 2/15 1/15 7/30 4/15 1/10 1/3 1/5 7/30 1/30 1/5 7/30
        2/15 4/15 1/10 1/30 7/30 2/15 1/15 1/30 3/10 1/3 1/5 1/10 2/15 1/30 2/15 4/15 0 4/15 1/5 4/15 1/10 1/10 1/3 7/30 3/10 1/3 3/10
        1/5 1/3 3/10 1/10 1/15 1/10 1/30 1/5 2/15 7/30 1/3 2/15 1/10 1/6 3/10 1/5 7/30 1/30 0 1/30 1/15 2/15 1/6 7/30 4/15 4/15 7/30
    ]

    upper1 = [
        7/15 17/30 13/30 3/5 17/30 17/30 17/30 13/30 3/5 2/3 11/30 7/15 0 1/2 17/30 13/30 7/15 13/30 17/30 13/30 2/5 2/5 2/3 2/5 17/30 2/5 19/30
        8/15 1/2 3/5 7/15 8/15 17/30 2/3 17/30 11/30 7/15 19/30 19/30 13/15 1/2 17/30 13/30 3/5 11/30 8/15 7/15 7/15 13/30 8/15 2/5 8/15 17/30 3/5
        11/30 1/3 2/5 8/15 7/15 3/5 2/3 17/30 2/3 8/15 2/15 3/5 2/3 3/5 17/30 2/3 7/15 8/15 2/5 2/5 11/30 17/30 17/30 1/2 2/5 19/30 13/30
    ]
    upper2 = [
        2/5 17/30 3/5 11/30 3/5 7/15 19/30 2/5 3/5 2/3 2/3 8/15 8/15 19/30 8/15 8/15 13/30 13/30 13/30 17/30 17/30 13/30 11/30 19/30 8/15 2/5 8/15
        1/3 13/30 11/30 2/5 2/3 2/3 0 13/30 1/2 17/30 17/30 1/3 2/5 1/3 13/30 11/30 8/15 1/3 1/2 8/15 8/15 8/15 8/15 2/5 3/5 2/3 13/30
        17/30 3/5 8/15 1/2 7/15 1/2 2/3 17/30 11/30 2/5 1/2 7/15 2/5 17/30 11/30 2/5 11/30 2/3 1/3 2/3 17/30 8/15 17/30 3/5 2/5 19/30 11/30
    ]
    upper3 = [
        3/5 17/30 1/2 3/5 19/30 2/5 8/15 1/3 11/30 2/5 17/30 13/30 2/5 3/5 3/5 11/30 1/2 11/30 2/3 17/30 3/5 7/15 19/30 1/2 3/5 1/3 19/30
        3/5 2/3 13/30 19/30 1/3 2/5 17/30 7/15 11/30 3/5 19/30 7/15 2/5 8/15 17/30 11/30 19/30 13/30 2/3 17/30 8/15 13/30 13/30 3/5 1/2 8/15 8/15
        3/5 2/3 1/2 1/2 2/3 7/15 3/5 3/5 1/2 1/3 2/5 8/15 2/5 11/30 1/3 8/15 7/15 13/30 0 2/5 11/30 19/30 19/30 2/5 1/2 7/15 7/15
    ]

    prob = IntervalMDP.cu(OrthogonalIntervalProbabilities(
        (
            IntervalProbabilities(; lower = lower1, upper = upper1),
            IntervalProbabilities(; lower = lower2, upper = upper2),
            IntervalProbabilities(; lower = lower3, upper = upper3),
        ),
        (Int32(3), Int32(3), Int32(3)),
    ))

    V = [
        23,
        27,
        16,
        6,
        26,
        17,
        12,
        9,
        8,
        22,
        1,
        21,
        11,
        24,
        4,
        10,
        13,
        19,
        3,
        14,
        25,
        20,
        18,
        7,
        5,
        15,
        2,
    ]
    V = reshape(Float64.(V), 3, 3, 3)
    V = IntervalMDP.cu(V)

    #### Maximization
    @testset "maximization" begin
        V111_expected = 17.276
        V222_expected = 18.838037037037
        V333_expected = 17.3777407407407
        V131_expected = 18.8653703703704
        V122_expected = 20.52
        V113_expected = 19.5096296296296

        Vres_first = bellman(V, prob; upper_bound = true)
        Vres_first = IntervalMDP.cpu(Vres_first)
        @test all(Vres_first .>= 0.0)
        @test Vres_first[1, 1, 1] ≈ V111_expected
        @test Vres_first[2, 2, 2] ≈ V222_expected
        @test Vres_first[3, 3, 3] ≈ V333_expected
        @test Vres_first[1, 3, 1] ≈ V131_expected
        @test Vres_first[1, 2, 2] ≈ V122_expected
        @test Vres_first[1, 1, 3] ≈ V113_expected

        # Vres = similar(V, 3, 3, 3)
        # bellman!(Vres, V, prob; upper_bound = true)
        # Vres = IntervalMDP.cpu(Vres)
        # @test Vres ≈ Vres_first

        # ws = construct_workspace(prob)
        # strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
        # Vres = similar(V, 3, 3, 3)
        # bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = true)
        # Vres = IntervalMDP.cpu(Vres)
        # @test Vres ≈ Vres_first
    end

    #### Minimization
    @testset "minimization" begin
        V111_expected = 10.1567407407407
        V222_expected = 10.4691111111111
        V333_expected = 11.4640740740741
        V131_expected = 7.8724444444445
        V122_expected = 10.3088888888889
        V113_expected = 11.5774074074074

        Vres_first = bellman(V, prob; upper_bound = false)
        Vres_first = IntervalMDP.cpu(Vres_first)
        @test all(Vres_first .>= 0.0)
        @test Vres_first[1, 1, 1] ≈ V111_expected
        @test Vres_first[2, 2, 2] ≈ V222_expected
        @test Vres_first[3, 3, 3] ≈ V333_expected
        @test Vres_first[1, 3, 1] ≈ V131_expected
        @test Vres_first[1, 2, 2] ≈ V122_expected
        @test Vres_first[1, 1, 3] ≈ V113_expected

        # Vres = similar(V, 3, 3, 3)
        # bellman!(Vres, V, prob; upper_bound = false)
        # Vres = IntervalMDP.cpu(Vres)
        # @test Vres ≈ Vres_first

        # ws = construct_workspace(prob)
        # strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
        # Vres = similar(V, 3, 3, 3)
        # bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = false)
        # Vres = IntervalMDP.cpu(Vres)
        # @test Vres ≈ Vres_first
    end
end

# @testset "Orthogonal abstraction" begin
#     using LazySets
#     using SpecialFunctions: erf
#     using LinearAlgebra: I

#     function IMDP_orthogonal_abstraction()
#         A = 0.9 * I
#         B = 0.7 * I
#         sigma = 2.0

#         X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
#         X1 = Interval(-10.0, 10.0)
#         X2 = Interval(-10.0, 10.0)
#         U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])

#         reach_region = Hyperrectangle(; low = [4.0, -6.0], high = [10.0, -2.0])

#         l = [5, 5]
#         X1_split = split(X1, l[1])
#         X2_split = split(X2, l[2])

#         X_split = Matrix{LazySet}(undef, l[1], l[2])
#         for j in 1:l[2]
#             for i in 1:l[1]
#                 x1 = X1_split[i]
#                 x2 = X2_split[j]
#                 X_split[i, j] = Hyperrectangle(
#                     [center(x1)[1], center(x2)[1]],
#                     [radius_hyperrectangle(x1)[1], radius_hyperrectangle(x2)[1]],
#                 )
#             end
#         end

#         U_split = split(U, [3, 3])

#         transition_prob(x, v_lower, v_upper) =
#             0.5 *
#             erf((x - v_upper) / (sigma * sqrt(2.0)), (x - v_lower) / (sigma * sqrt(2.0)))

#         probs1 = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
#         probs2 = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
#         stateptr = Int32[1]

#         for source2 in 1:(l[2] + 1)
#             for source1 in 1:(l[1] + 1)
#                 if source1 == 1 || source2 == 1
#                     probs1_lower = zeros(l[1] + 1, 1)
#                     probs1_upper = zeros(l[1] + 1, 1)

#                     probs1_upper[source1, 1] = 1
#                     probs1_lower[source1, 1] = 1

#                     probs2_lower = zeros(l[2] + 1, 1)
#                     probs2_upper = zeros(l[2] + 1, 1)

#                     probs2_upper[source2, 1] = 1
#                     probs2_lower[source2, 1] = 1

#                     push!(
#                         probs1,
#                         IntervalProbabilities(; lower = probs1_lower, upper = probs1_upper),
#                     )
#                     push!(
#                         probs2,
#                         IntervalProbabilities(; lower = probs2_lower, upper = probs2_upper),
#                     )
#                 else
#                     Xij = X_split[source1 - 1, source2 - 1]

#                     for u in U_split
#                         Xij_u = A * Xij + B * u
#                         Xij_u = box_approximation(Xij_u)

#                         probs1_lower = zeros(l[1] + 1, 1)
#                         probs1_upper = zeros(l[1] + 1, 1)

#                         for target1 in 1:(l[1] + 1)
#                             if target1 == 1
#                                 probs1_upper[target1, 1] =
#                                     max(
#                                         1 - transition_prob(
#                                             low(Xij_u)[1],
#                                             low(X)[1],
#                                             high(X)[1],
#                                         ),
#                                         1 - transition_prob(
#                                             high(Xij_u)[1],
#                                             low(X)[1],
#                                             high(X)[1],
#                                         ),
#                                     ) + eps(Float64)
#                                 probs1_lower[target1, 1] = min(
#                                     1 - transition_prob(
#                                         center(Xij_u)[1],
#                                         low(X)[1],
#                                         high(X)[1],
#                                     ),
#                                     1 - transition_prob(
#                                         low(Xij_u)[1],
#                                         low(X)[1],
#                                         high(X)[1],
#                                     ),
#                                     1 - transition_prob(
#                                         high(Xij_u)[1],
#                                         low(X)[1],
#                                         high(X)[1],
#                                     ),
#                                 )
#                             else
#                                 probs1_upper[target1, 1] = max(
#                                     transition_prob(
#                                         center(Xij_u)[1],
#                                         low(X1_split[target1 - 1])[1],
#                                         high(X1_split[target1 - 1])[1],
#                                     ),
#                                     transition_prob(
#                                         low(Xij_u)[1],
#                                         low(X1_split[target1 - 1])[1],
#                                         high(X1_split[target1 - 1])[1],
#                                     ),
#                                     transition_prob(
#                                         high(Xij_u)[1],
#                                         low(X1_split[target1 - 1])[1],
#                                         high(X1_split[target1 - 1])[1],
#                                     ),
#                                 )
#                                 probs1_lower[target1, 1] = min(
#                                     transition_prob(
#                                         low(Xij_u)[1],
#                                         low(X1_split[target1 - 1])[1],
#                                         high(X1_split[target1 - 1])[1],
#                                     ),
#                                     transition_prob(
#                                         high(Xij_u)[1],
#                                         low(X1_split[target1 - 1])[1],
#                                         high(X1_split[target1 - 1])[1],
#                                     ),
#                                 )
#                             end
#                         end

#                         probs2_lower = zeros(l[2] + 1, 1)
#                         probs2_upper = zeros(l[2] + 1, 1)

#                         for target2 in 1:(l[2] + 1)
#                             if target2 == 1
#                                 probs2_upper[target2, 1] =
#                                     max(
#                                         1 - transition_prob(
#                                             low(Xij_u)[2],
#                                             low(X)[2],
#                                             high(X)[2],
#                                         ),
#                                         1 - transition_prob(
#                                             high(Xij_u)[2],
#                                             low(X)[2],
#                                             high(X)[2],
#                                         ),
#                                     ) + eps(Float64)
#                                 probs2_lower[target2, 1] = min(
#                                     1 - transition_prob(
#                                         center(Xij_u)[2],
#                                         low(X)[2],
#                                         high(X)[2],
#                                     ),
#                                     1 - transition_prob(
#                                         low(Xij_u)[2],
#                                         low(X)[2],
#                                         high(X)[2],
#                                     ),
#                                     1 - transition_prob(
#                                         high(Xij_u)[2],
#                                         low(X)[2],
#                                         high(X)[2],
#                                     ),
#                                 )
#                             else
#                                 probs2_upper[target2, 1] = max(
#                                     transition_prob(
#                                         center(Xij_u)[2],
#                                         low(X2_split[target2 - 1])[1],
#                                         high(X2_split[target2 - 1])[1],
#                                     ),
#                                     transition_prob(
#                                         low(Xij_u)[2],
#                                         low(X2_split[target2 - 1])[1],
#                                         high(X2_split[target2 - 1])[1],
#                                     ),
#                                     transition_prob(
#                                         high(Xij_u)[2],
#                                         low(X2_split[target2 - 1])[1],
#                                         high(X2_split[target2 - 1])[1],
#                                     ),
#                                 )
#                                 probs2_lower[target2, 1] = min(
#                                     transition_prob(
#                                         low(Xij_u)[2],
#                                         low(X2_split[target2 - 1])[1],
#                                         high(X2_split[target2 - 1])[1],
#                                     ),
#                                     transition_prob(
#                                         high(Xij_u)[2],
#                                         low(X2_split[target2 - 1])[1],
#                                         high(X2_split[target2 - 1])[1],
#                                     ),
#                                 )
#                             end
#                         end

#                         push!(
#                             probs1,
#                             IntervalProbabilities(;
#                                 lower = probs1_lower,
#                                 upper = probs1_upper,
#                             ),
#                         )
#                         push!(
#                             probs2,
#                             IntervalProbabilities(;
#                                 lower = probs2_lower,
#                                 upper = probs2_upper,
#                             ),
#                         )
#                     end
#                 end

#                 push!(stateptr, length(probs1) + 1)
#             end
#         end

#         probs1, _ = IntervalMDP.interval_prob_hcat(probs1)
#         probs2, _ = IntervalMDP.interval_prob_hcat(probs2)
#         probs = OrthogonalIntervalProbabilities(
#             (probs1, probs2),
#             (Int32(l[1] + 1), Int32(l[2] + 1)),
#         )
#         pmdp = IntervalMDP.cu(OrthogonalIntervalMarkovDecisionProcess(probs, stateptr))

#         reach = Tuple{Int32, Int32}[]
#         avoid = Tuple{Int32, Int32}[]

#         for j in 1:(l[2] + 1)
#             for i in 1:(l[1] + 1)
#                 if j == 1 || i == 1
#                     push!(avoid, (i, j))
#                 elseif X_split[i - 1, j - 1] ⊆ reach_region
#                     push!(reach, (i, j))
#                 end
#             end
#         end

#         return pmdp, reach, avoid
#     end

#     function IMDP_direct_abstraction()
#         A = 0.9I(2)
#         B = 0.7I(2)
#         sigma = 2.0

#         X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
#         X1 = Interval(-10.0, 10.0)
#         X2 = Interval(-10.0, 10.0)
#         U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])

#         reach_region = Hyperrectangle(; low = [4.0, -6.0], high = [10.0, -2.0])

#         l = [5, 5]
#         X1_split = split(X1, l[1])
#         X2_split = split(X2, l[2])

#         X_split = Matrix{LazySet}(undef, l[1] + 1, l[2] + 1)
#         for j in 1:(l[2] + 1)
#             for i in 1:(l[1] + 1)
#                 if i == 1 && j == 1
#                     X_split[i, j] = CartesianProduct(
#                         Complement(Interval(low(X, 1), high(X, 1))),
#                         Complement(Interval(low(X, 2), high(X, 2))),
#                     )
#                 elseif i == 1
#                     x2 = X2_split[j - 1]
#                     X_split[i, j] = CartesianProduct(
#                         Complement(Interval(low(X, 1), high(X, 1))),
#                         Interval(low(x2, 1), high(x2, 1)),
#                     )
#                 elseif j == 1
#                     x1 = X1_split[i - 1]
#                     X_split[i, j] = CartesianProduct(
#                         Interval(low(x1, 1), high(x1, 1)),
#                         Complement(Interval(low(X, 2), high(X, 2))),
#                     )
#                 else
#                     x1 = X1_split[i - 1]
#                     x2 = X2_split[j - 1]
#                     X_split[i, j] = Hyperrectangle(
#                         [center(x1)[1], center(x2)[1]],
#                         [radius_hyperrectangle(x1)[1], radius_hyperrectangle(x2)[1]],
#                     )
#                 end
#             end
#         end

#         U_split = split(U, [3, 3])

#         transition_prob(x, v_lower, v_upper) =
#             0.5 *
#             erf((x - v_upper) / (sigma * sqrt(2.0)), (x - v_lower) / (sigma * sqrt(2.0)))

#         probs = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
#         for source2 in 1:(l[2] + 1)
#             for source1 in 1:(l[1] + 1)
#                 source = (source2 - 1) * (l[1] + 1) + source1

#                 probs_lower = Vector{Float64}[]
#                 probs_upper = Vector{Float64}[]

#                 if source1 == 1 || source2 == 1
#                     prob_upper = zeros(prod(l .+ 1))
#                     prob_lower = zeros(prod(l .+ 1))

#                     prob_upper[source] = 1
#                     prob_lower[source] = 1

#                     push!(probs_lower, prob_lower)
#                     push!(probs_upper, prob_upper)
#                 else
#                     Xij = X_split[source1, source2]

#                     for u in U_split
#                         Xij_u = A * Xij + B * u
#                         box_Xij_u = box_approximation(Xij_u)

#                         prob_upper = zeros(prod(l .+ 1))
#                         prob_lower = zeros(prod(l .+ 1))

#                         for target2 in 1:(l[2] + 1)
#                             for target1 in 1:(l[1] + 1)
#                                 Xij_target = X_split[target1, target2]
#                                 target = (target2 - 1) * (l[1] + 1) + target1

#                                 if target1 == 1 && target2 == 1
#                                     prob_upper[target] =
#                                         max(
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                         ) * max(
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                         )
#                                     prob_lower[target] =
#                                         min(
#                                             1 - transition_prob(
#                                                 center(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                         ) * min(
#                                             1 - transition_prob(
#                                                 center(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                         )
#                                 elseif target1 == 1
#                                     prob_upper[target] =
#                                         max(
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                         ) * max(
#                                             transition_prob(
#                                                 center(box_Xij_u)[2],
#                                                 low(Xij_target.Y)[1],
#                                                 high(Xij_target.Y)[1],
#                                             ),
#                                             transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(Xij_target.Y)[1],
#                                                 high(Xij_target.Y)[1],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(Xij_target.Y)[1],
#                                                 high(Xij_target.Y)[1],
#                                             ),
#                                         )
#                                     prob_lower[target] =
#                                         min(
#                                             1 - transition_prob(
#                                                 center(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                         ) * min(
#                                             transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(Xij_target.Y)[1],
#                                                 high(Xij_target.Y)[1],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(Xij_target.Y)[1],
#                                                 high(Xij_target.Y)[1],
#                                             ),
#                                         )
#                                 elseif target2 == 1
#                                     prob_upper[target] =
#                                         max(
#                                             transition_prob(
#                                                 center(box_Xij_u)[1],
#                                                 low(Xij_target.X)[1],
#                                                 high(Xij_target.X)[1],
#                                             ),
#                                             transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(Xij_target.X)[1],
#                                                 high(Xij_target.X)[1],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(Xij_target.X)[1],
#                                                 high(Xij_target.X)[1],
#                                             ),
#                                         ) * max(
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                         )
#                                     prob_lower[target] =
#                                         min(
#                                             transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(Xij_target.X)[1],
#                                                 high(Xij_target.X)[1],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(Xij_target.X)[1],
#                                                 high(Xij_target.X)[1],
#                                             ),
#                                         ) * min(
#                                             1 - transition_prob(
#                                                 center(box_Xij_u)[2],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                         )
#                                 else
#                                     prob_upper[target] =
#                                         max(
#                                             transition_prob(
#                                                 center(box_Xij_u)[1],
#                                                 low(Xij_target)[1],
#                                                 high(Xij_target)[1],
#                                             ),
#                                             transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(Xij_target)[1],
#                                                 high(Xij_target)[1],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(Xij_target)[1],
#                                                 high(Xij_target)[1],
#                                             ),
#                                         ) * max(
#                                             transition_prob(
#                                                 center(box_Xij_u)[2],
#                                                 low(Xij_target)[2],
#                                                 high(Xij_target)[2],
#                                             ),
#                                             transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(Xij_target)[2],
#                                                 high(Xij_target)[2],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(Xij_target)[2],
#                                                 high(Xij_target)[2],
#                                             ),
#                                         )
#                                     prob_lower[target] =
#                                         min(
#                                             transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(Xij_target)[1],
#                                                 high(Xij_target)[1],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(Xij_target)[1],
#                                                 high(Xij_target)[1],
#                                             ),
#                                         ) * min(
#                                             transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(Xij_target)[2],
#                                                 high(Xij_target)[2],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(Xij_target)[2],
#                                                 high(Xij_target)[2],
#                                             ),
#                                         )
#                                 end
#                             end
#                         end

#                         push!(probs_lower, prob_lower)
#                         push!(probs_upper, prob_upper)
#                     end
#                 end

#                 prob = IntervalProbabilities(;
#                     lower = reduce(hcat, probs_lower),
#                     upper = reduce(hcat, probs_upper),
#                 )
#                 push!(probs, prob)
#             end
#         end
#         mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(probs))

#         reach = Int32[]
#         avoid = Int32[]

#         for source2 in 1:(l[2] + 1)
#             for source1 in 1:(l[1] + 1)
#                 Xij = X_split[source1, source2]
#                 source = (source2 - 1) * (l[1] + 1) + source1

#                 if source1 == 1 || source2 == 1
#                     push!(avoid, source)
#                 elseif Xij ⊆ reach_region
#                     push!(reach, source)
#                 end
#             end
#         end

#         return mdp, reach, avoid
#     end

#     # Orthogonal abstraction
#     pmdp, reach_set, avoid_set = IMDP_orthogonal_abstraction()

#     prop = FiniteTimeReachAvoid(reach_set, avoid_set, 10)
#     spec = Specification(prop, Pessimistic, Maximize)
#     prob_ortho = Problem(pmdp, spec)

#     V_ortho, it_ortho, res_ortho = value_iteration(prob_ortho)
#     @test all(V_ortho .≥ 0.0)

#     # Direct abstraction
#     mdp, reach_set, avoid_set = IMDP_direct_abstraction()

#     prop = FiniteTimeReachAvoid(reach_set, avoid_set, 10)
#     spec = Specification(prop, Pessimistic, Maximize)
#     prob_direct = Problem(mdp, spec)

#     V_direct, it_direct, res_direct = value_iteration(prob_direct)

#     @test it_ortho == it_direct
#     @test all(V_ortho .≥ reshape(V_direct, 6, 6))
# end

# # 3-D random
# @testset "3D random" begin
#     rng = MersenneTwister(995)

#     prob_lower = [rand(rng, Float64, 3, 27) ./ 3.0 for _ in 1:3]
#     prob_upper = [(rand(rng, Float64, 3, 27) .+ 1.0) ./ 3.0 for _ in 1:3]

#     probs = OrthogonalIntervalProbabilities(
#         ntuple(
#             i -> IntervalProbabilities(; lower = prob_lower[i], upper = prob_upper[i]),
#             3,
#         ),
#         (Int32(3), Int32(3), Int32(3)),
#     )

#     mdp = IntervalMDP.cu(OrthogonalIntervalMarkovChain(probs))

#     prop = FiniteTimeReachability([(3, 3, 3)], 10)
#     spec = Specification(prop, Pessimistic, Maximize)
#     prob = Problem(mdp, spec)

#     V_ortho, it_ortho, res_ortho = value_iteration(prob)
#     V_ortho = IntervalMDP.cpu(V_ortho)

#     @test V_ortho[3, 3, 3] ≈ 1.0
#     @test all(V_ortho .>= 0.0)

#     # Test against the naive construction
#     prob_lower_simple = zeros(27, 27)
#     prob_upper_simple = zeros(27, 27)

#     for i in 1:27
#         for (j₁, j₂, j₃) in Iterators.product(1:3, 1:3, 1:3)
#             j = (j₃ - 1) * 9 + (j₂ - 1) * 3 + j₁

#             prob_lower_simple[j, i] =
#                 prob_lower[1][j₁, i] * prob_lower[2][j₂, i] * prob_lower[3][j₃, i]
#             prob_upper_simple[j, i] =
#                 prob_upper[1][j₁, i] * prob_upper[2][j₂, i] * prob_upper[3][j₃, i]
#         end
#     end

#     probs = IntervalProbabilities(; lower = prob_lower_simple, upper = prob_upper_simple)

#     mdp = IntervalMarkovChain(probs)

#     prop = FiniteTimeReachability([27], 10)
#     spec = Specification(prop, Pessimistic, Maximize)
#     prob = Problem(mdp, spec)

#     V_direct, it_direct, res_direct = value_iteration(prob)
#     @test V_direct[27] ≈ 1.0
#     @test all(V_ortho .≥ reshape(V_direct, 3, 3, 3))
# end

# @testset "synthesis" begin
#     rng = MersenneTwister(3286)

#     num_states_per_axis = 3
#     num_axis = 3
#     num_states = num_states_per_axis^num_axis
#     num_actions = 2
#     num_choices = num_states * num_actions

#     prob_lower = [
#         rand(rng, Float64, num_states_per_axis, num_choices) ./ num_states_per_axis for
#         _ in 1:num_axis
#     ]
#     prob_upper = [
#         (rand(rng, Float64, num_states_per_axis, num_choices) .+ 1.0) ./
#         num_states_per_axis for _ in 1:num_axis
#     ]

#     probs = OrthogonalIntervalProbabilities(
#         ntuple(
#             i -> IntervalProbabilities(; lower = prob_lower[i], upper = prob_upper[i]),
#             num_axis,
#         ),
#         (
#             Int32(num_states_per_axis),
#             Int32(num_states_per_axis),
#             Int32(num_states_per_axis),
#         ),
#     )

#     stateptr = [Int32[1]; convert.(Int32, 1 .+ collect(1:num_states) .* 2)]
#     mdp = IntervalMDP.cu(OrthogonalIntervalMarkovDecisionProcess(probs, stateptr))

#     prop = FiniteTimeReachability(
#         [(num_states_per_axis, num_states_per_axis, num_states_per_axis)],
#         10,
#     )
#     spec = Specification(prop, Pessimistic, Maximize)
#     prob = Problem(mdp, spec)

#     policy, V, it, res = control_synthesis(prob)
#     @test it == 10
#     @test all(V .≥ 0.0)

#     # Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
#     prob = Problem(mdp, spec, policy)
#     V_mc, k, res = value_iteration(prob)
#     @test V ≈ V_mc
# end
