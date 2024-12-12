using Revise, Test
using IntervalMDP, SparseArrays, CUDA
using StatsBase
using Random: MersenneTwister

#### Maximization
@testset "maximization" begin
    prob = IntervalMDP.cu(
        IntervalProbabilities(;
            lower = sparse_hcat(
                SparseVector(Int32(15), Int32[4, 10], [0.1, 0.2]),
                SparseVector(Int32(15), Int32[5, 6, 7], [0.5, 0.3, 0.1]),
            ),
            upper = sparse_hcat(
                SparseVector(Int32(15), Int32[1, 4, 10], [0.5, 0.6, 0.7]),
                SparseVector(Int32(15), Int32[5, 6, 7], [0.7, 0.5, 0.3]),
            ),
        ),
    )

    V = IntervalMDP.cu(collect(1.0:15.0))

    Vres = bellman(V, prob; upper_bound = true)
    Vres = IntervalMDP.cpu(Vres)
    @test Vres ≈ [0.3 * 4 + 0.7 * 10, 0.5 * 5 + 0.3 * 6 + 0.2 * 7]

    # To GPU first
    prob = IntervalProbabilities(;
        lower = IntervalMDP.cu(
            sparse_hcat(
                SparseVector(Int32(15), Int32[4, 10], [0.1, 0.2]),
                SparseVector(Int32(15), Int32[5, 6, 7], [0.5, 0.3, 0.1]),
            ),
        ),
        upper = IntervalMDP.cu(
            sparse_hcat(
                SparseVector(Int32(15), Int32[1, 4, 10], [0.5, 0.6, 0.7]),
                SparseVector(Int32(15), Int32[5, 6, 7], [0.7, 0.5, 0.3]),
            ),
        ),
    )

    V = IntervalMDP.cu(collect(1.0:15.0))

    Vres = bellman(V, prob; upper_bound = true)
    Vres = IntervalMDP.cpu(Vres)
    @test Vres ≈ [0.3 * 4 + 0.7 * 10, 0.5 * 5 + 0.3 * 6 + 0.2 * 7]

    ws = construct_workspace(prob)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(V, 2)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = true)
    Vres = IntervalMDP.cpu(Vres)
    @test Vres ≈ [0.3 * 4 + 0.7 * 10, 0.5 * 5 + 0.3 * 6 + 0.2 * 7]
end

#### Minimization
@testset "minimization" begin
    prob = IntervalMDP.cu(
        IntervalProbabilities(;
            lower = sparse_hcat(
                SparseVector(Int32(15), Int32[4, 10], [0.1, 0.2]),
                SparseVector(Int32(15), Int32[5, 6, 7], [0.5, 0.3, 0.1]),
            ),
            upper = sparse_hcat(
                SparseVector(Int32(15), Int32[1, 4, 10], [0.5, 0.6, 0.7]),
                SparseVector(Int32(15), Int32[5, 6, 7], [0.7, 0.5, 0.3]),
            ),
        ),
    )

    V = IntervalMDP.cu(collect(1.0:15.0))

    Vres = bellman(V, prob; upper_bound = false)
    Vres = IntervalMDP.cpu(Vres)
    @test Vres ≈ [0.5 * 1 + 0.3 * 4 + 0.2 * 10, 0.6 * 5 + 0.3 * 6 + 0.1 * 7]

    ws = construct_workspace(prob)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(V, 2)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = false)
    Vres = IntervalMDP.cpu(Vres)
    @test Vres ≈ [0.5 * 1 + 0.3 * 4 + 0.2 * 10, 0.6 * 5 + 0.3 * 6 + 0.1 * 7]
end

#### Large matrices
@testset "large matrices" begin
    function sample_sparse_interval_probabilities(rng, n, m, nnz_per_column)
        prob_split = 1 / nnz_per_column

        rand_val_lower = rand(rng, Float64, nnz_per_column * m) .* prob_split
        rand_val_upper = rand(rng, Float64, nnz_per_column * m) .* prob_split .+ prob_split
        rand_index = collect(1:nnz_per_column)

        row_vals = Vector{Int32}(undef, nnz_per_column * m)
        col_ptrs = Int32[1; collect(1:m) .* nnz_per_column .+ 1]

        for j in 1:m
            StatsBase.seqsample_a!(rng, 1:n, rand_index)  # Select nnz_per_column elements from 1:n
            sort!(rand_index)

            row_vals[((j - 1) * nnz_per_column + 1):(j * nnz_per_column)] .= rand_index
        end

        lower = SparseMatrixCSC{Float64, Int32}(n, m, col_ptrs, row_vals, rand_val_lower)
        upper = SparseMatrixCSC{Float64, Int32}(n, m, col_ptrs, row_vals, rand_val_upper)

        prob = IntervalProbabilities(; lower = lower, upper = upper)
        V = rand(rng, Float64, n)

        cuda_prob = IntervalMDP.cu(prob)
        cuda_V = IntervalMDP.cu(V)

        return prob, V, cuda_prob, cuda_V
    end

    # Many columns
    @testset "many columns" begin
        rng = MersenneTwister(55392)

        n = 100
        m = 1000000  # It has to be greater than 8 * 2^16 to exceed maximum grid size
        nnz_per_column = 2
        prob, V, cuda_prob, cuda_V =
            sample_sparse_interval_probabilities(rng, n, m, nnz_per_column)

        V_cpu = bellman(V, prob; upper_bound = false)
        V_gpu = IntervalMDP.cpu(bellman(cuda_V, cuda_prob; upper_bound = false))

        @test V_cpu ≈ V_gpu
    end

    # Many non-zeros
    @testset "many non-zeros" begin
        rng = MersenneTwister(55392)

        n = 100000
        m = 10
        nnz_per_column = 1500   # It has to be greater than 187 to fill shared memory with 8 states per block.
        prob, V, cuda_prob, cuda_V =
            sample_sparse_interval_probabilities(rng, n, m, nnz_per_column)

        V_cpu = bellman(V, prob; upper_bound = false)
        V_gpu = IntervalMDP.cpu(bellman(cuda_V, cuda_prob; upper_bound = false))

        @test V_cpu ≈ V_gpu
    end

    # More non-zeros
    @testset "more non-zeros" begin
        rng = MersenneTwister(55392)

        n = 100000
        m = 10
        nnz_per_column = 4000   # It has to be greater than 3800 to exceed shared memory for ff implementation
        prob, V, cuda_prob, cuda_V =
            sample_sparse_interval_probabilities(rng, n, m, nnz_per_column)

        V_cpu = bellman(V, prob; upper_bound = false)
        V_gpu = IntervalMDP.cpu(bellman(cuda_V, cuda_prob; upper_bound = false))

        @test V_cpu ≈ V_gpu
    end

    # Most non-zeros
    @testset "most non-zeros" begin
        rng = MersenneTwister(55392)

        n = 100000
        m = 10
        nnz_per_column = 6000   # It has to be greater than 5800 to exceed shared memory for fi implementation
        prob, V, cuda_prob, cuda_V =
            sample_sparse_interval_probabilities(rng, n, m, nnz_per_column)

        V_cpu = bellman(V, prob; upper_bound = false)
        V_gpu = IntervalMDP.cpu(bellman(cuda_V, cuda_prob; upper_bound = false))

        @test V_cpu ≈ V_gpu
    end

    # Too many non-zeros
    @testset "too many non-zeros" begin
        rng = MersenneTwister(55392)

        n = 100000
        m = 10
        nnz_per_column = 8000   # It has to be greater than 7800 to exceed shared memory for ii implementation
        prob, V, cuda_prob, cuda_V =
            sample_sparse_interval_probabilities(rng, n, m, nnz_per_column)

        @test_throws IntervalMDP.OutOfSharedMemory bellman(
            cuda_V,
            cuda_prob;
            upper_bound = false,
        )
    end
end
