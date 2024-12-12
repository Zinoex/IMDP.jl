function IntervalMDP.bellman!(
    workspace::CuDenseWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    stateptr;
    upper_bound = false,
    maximize = true,
) where {Tv}
    max_states_per_block = 32
    shmem(threads) = begin
        threads = prevwarp(device(), threads)
        num_warps = div(threads, 32)

        shmem = sum(size(V)) * num_warps * (sizeof(Int32) + sizeof(Tv)) +
        max_states_per_block * workspace.max_actions * sizeof(Tv)

        return shmem
    end

    kernel = @cuda launch = false dense_bellman_kernel!(
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        prob,
        stateptr,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv)),
    )

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    # Execution plan:
    # - value assignment: 1 warp per state
    # - squeeze as many states as possible in a block
    # - use shared memory to store the values and permutation
    # - use bitonic sort to sort the values for all states in a block
    wanted_threads = min(1024, 32 * length(Vres))

    threads = min(max_threads, wanted_threads)
    warps = div(threads, 32)
    blocks = min(2^16 - 1, cld(length(Vres), warps))

    kernel(
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        prob,
        stateptr,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv));
        blocks = blocks,
        threads = threads,
        shmem = shmem,
    )

    return Vres
end

function dense_bellman_kernel!(
    workspace,
    strategy_cache,
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    stateptr,
    value_lt,
    action_reduce,
) where {Tv}
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    # Prepare action workspace shared memory
    action_workspace = CuDynamicSharedArray(Tv, (workspace.max_actions, nwarps))
    @inbounds action_workspace = @view action_workspace[:, wid]

    # Prepare sorting shared memory
    value = CuDynamicSharedArray(Tv, length(V), nwarps * workspace.max_actions * sizeof(Tv))
    perm = CuDynamicSharedArray(
        Int32,
        length(V),
        (nwarps * workspace.max_actions + length(V)) * sizeof(Tv),
    )

    # Perform sorting
    dense_initialize_sorting_shared_memory!(V, value, perm)
    block_bitonic_sort!(value, perm, value_lt)

    # O-maxmization
    dense_omaximization!(
        action_workspace,
        strategy_cache,
        Vres,
        value,
        perm,
        prob,
        stateptr,
        action_reduce,
    )

    return nothing
end

@inline function dense_initialize_sorting_shared_memory!(V, value, perm)
    # Copy into shared memory
    i = threadIdx().x
    @inbounds while i <= length(V)
        value[i] = V[i]
        perm[i] = i
        i += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

@inline function dense_omaximization!(
    action_workspace,
    strategy_cache,
    Vres,
    value,
    perm,
    prob,
    stateptr,
    action_reduce,
)
    assume(warpsize() == 32)

    warps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    j = wid + (blockIdx().x - one(Int32)) * warps
    @inbounds while j <= length(Vres)
        state_dense_omaximization!(
            action_workspace,
            strategy_cache,
            Vres,
            value,
            perm,
            prob,
            stateptr,
            action_reduce,
            j,
        )
        j += gridDim().x * warps
    end

    return nothing
end

@inline function state_dense_omaximization!(
    action_workspace,
    strategy_cache::OptimizingActiveCache,
    Vres,
    value,
    perm,
    prob::IntervalProbabilities{Tv},
    stateptr,
    action_reduce,
    jₛ,
) where {Tv}
    lane = mod1(threadIdx().x, warpsize())

    s₁, s₂ = stateptr[jₛ], stateptr[jₛ + one(Int32)]
    nactions = s₂ - s₁
    @inbounds action_values = @view action_workspace[1:nactions]

    k = one(Int32)
    @inbounds while k <= nactions
        jₐ = s₁ + k - one(Int32)
        lowerⱼ = @view lower(prob)[:, jₐ]
        gapⱼ = @view gap(prob)[:, jₐ]
        sum_lowerⱼ = sum_lower(prob)[jₐ]

        # Use O-maxmization to find the value for the action
        v = state_action_dense_omaximization!(value, perm, lowerⱼ, gapⱼ, sum_lowerⱼ, lane)

        if lane == one(Int32)
            action_values[k] = v
        end
        sync_warp()

        k += one(Int32)
    end

    # Find the best action
    v = extract_strategy_warp!(strategy_cache, action_values, Vres, jₛ, action_reduce, lane)

    if lane == one(Int32)
        Vres[jₛ] = v
    end
    sync_warp()
end

@inline function state_dense_omaximization!(
    action_workspace,
    strategy_cache::NonOptimizingActiveCache,
    Vres,
    value,
    perm,
    prob::IntervalProbabilities{Tv},
    stateptr,
    action_reduce,
    jₛ,
) where {Tv}
    lane = mod1(threadIdx().x, warpsize())

    @inbounds begin
        s₁ = stateptr[jₛ]
        jₐ = s₁ + strategy_cache[jₛ] - one(Int32)
        lowerⱼ = @view lower(prob)[:, jₐ]
        gapⱼ = @view gap(prob)[:, jₐ]
        sum_lowerⱼ = sum_lower(prob)[jₐ]

        # Use O-maxmization to find the value for the action
        v = state_action_dense_omaximization!(value, perm, lowerⱼ, gapⱼ, sum_lowerⱼ, lane)

        if lane == one(Int32)
            Vres[jₛ] = v
        end
        sync_warp()
    end
end

@inline function state_action_dense_omaximization!(
    value,
    perm,
    lower,
    gap,
    sum_lower::Tv,
    lane,
) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(lower))
    remaining = one(Tv) - sum_lower
    gap_value = zero(Tv)

    # Add the lower bound multiplied by the value
    s = lane
    @inbounds while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding lower bound and multipy by the value
        if s <= length(lower)
            p = perm[s]

            gap_value += lower[p] * value[s]
        end

        s += warpsize()
    end

    # Add the gap multiplied by the value
    s = lane
    @inbounds while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= length(gap)
            gap[perm[s]]
        else
            # 0 gap is a neural element
            zero(Tv)
        end

        # Cummulatively sum the gap with a tree reduction
        cum_gap = cumsum_warp(g, lane)

        # Update the remaining probability
        remaining -= cum_gap
        remaining += g

        # Update the probability
        if s <= length(gap)
            g = clamp(remaining, zero(Tv), g)
            gap_value += g * value[s]
            remaining -= g
        end

        # Update the remaining probability from the last thread in the warp
        remaining = shfl_sync(0xffffffff, remaining, warpsize())

        # Early exit if the remaining probability is zero
        if remaining <= zero(Tv)
            break
        end

        s += warpsize()
    end

    gap_value = CUDA.reduce_warp(+, gap_value)
    return gap_value
end
