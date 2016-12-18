### Monte Carlo simulation of grand canonical ensemble of model in N_DIMS
using DataFrames
using Base.Test
using Gadfly
include("utils.jl")  # EnergyParams type defined here.

# Here, we write functions to compute the energy of a *single* linker and of a *single* adsorbate. 
#   These are useful during Monte Carlo simulations when we seek to propose:
#      (i) to rotate a linker 
#      (ii) to insert/remove an adsorbate
#   For these proposals, we need the energy difference in the system, which are given by these functions.

# VARIABLE CONVENTIONS
#    The lattice is a hypercube of dimension `N_DIMS`. Each side is of length `n_sites`.
#    The status of the occupancy of the lattice is stored in a multi-dimensional array `occupancy`.
#        0: unoccupied, 1: occupied
#        e.g. occupancy[4, 5] = 1 in 2D means that site [4, 5] is occupied.
#    There are rotating ligands (=linkers) on each of the N_DIMS faces of the hypercube.
#    The states of the ligands are stored in a N_DIMS dimensional array of multidimensional arrays.
#        0: parallel to cage walls, 1: rotated and sticking into pore
#        e.g. linker_states[2][4, 5] = 1 in 2D means that the ligand that interacts with adsorbates in the x-direction (1)
#          and on the face of the adsorption cage [4, 5] is rotated into the pores.
#    So there are N_DIMS * n_sites ^ N_DIMS total linkers and n_sites ^ N_DIMS total adsorption cages in the simulation.
"""
Compute the energy of linker in linker_states[which_face][which_cage...]
This includes intrahost penalty and guest-host interaction.

Paramters:
    which_face: for adsorbates in which of the N_DIMS direction does the linker interact?
    which_cage: the index of the cage to which the linker belongs e.g. [1, 4, 2] in 3D
    linker_states: see VARIABLE CONVENTIONS above
    occupancy: see VARIABLE CONVENTIONS above
    energy_params: energetic parameters of model
"""
function linker_energy(which_face::Int, which_cage::Array{Int, 1}, linker_states::Array{Array{Int, N_DIMS}}, occupancy::Array{Int, N_DIMS}, energy_params::EnergyParams)
    energy::Float64 = 0.0
    # intra-host energy. Linker is penalized if sticking into pore
    energy += energy_params.ϵ_l * linker_states[which_face][which_cage...]
    # guest-host energy.
    # We get a benefit if the linker is sticking into the pore AND an adsorbate is to the left/right.
    if linker_states[which_face][which_cage...] == 1 # if sticking into pore, assess if there is a benefit
        energy -= occupancy[which_cage...] * energy_params.ϵ # adsorbate to right/above of linker (i.e. same cage as linker)
        # adsorbate to left/below linker, deal with periodic boundary conditions here.
        id_left = deepcopy(which_cage)
        if which_cage[which_face] == 1
            # periodic boundary condition
            id_left[which_face] = size(occupancy)[1]
        else
            id_left[which_face] -= 1
        end
        energy -= occupancy[id_left...] * energy_params.ϵ
    end
    return energy
end

"""
Compute the energy of adsorbate in a given cage.
This depends on the configuration of the surrounding linkers.

Paramters:
    which_cage: the index of the cage where adsorbate of interest resides.
    linker_states: see VARIABLE CONVENTIONS above
    occupancy: see VARIABLE CONVENTIONS above
    energy_params: energetic parameters of model
"""
function adsorbate_energy(which_cage::Array{Int, 1}, linker_states::Array{Array{Int, N_DIMS}}, occupancy::Array{Int, N_DIMS}, energy_params::EnergyParams)
    energy::Float64 = 0.0
    if occupancy[which_cage...] == 1
        # host-adsorbate background energy, regardless of surrounding linker configuration
        energy -= energy_params.ϵ_0
        # additional energetic benefits due to linker sticking into pore
        for which_face = 1:N_DIMS # loop over faces of this cage
            # linker in same cage, i.e. to the left/below (gives zero if linker_state[which_cage] == 0)
            energy -= linker_states[which_face][which_cage...] * energy_params.ϵ
            # linker to right, need to apply periodic boundary conditions here
            id_right = deepcopy(which_cage)
            if which_cage[which_face] == size(occupancy)[1]
                id_right[which_face] = 1
            else
                id_right[which_face] += 1
            end
            energy -= linker_states[which_face][id_right...] * energy_params.ϵ
        end
    end
    return energy
end

# The following function computes the energy of the *entire* system of adsorbates instead of just a single adsorbate.
"""
Calculate the collective energy of each adsorbed guest molecule in the system.

Parameters:
    linker_states: see VARIABLE CONVENTIONS above
    occupancy: see VARIABLE CONVENTIONS above
    energy_params: energetic parameters of model
"""
function system_guest_energy(linker_states::Array{Array{Int, N_DIMS}}, occupancy::Array{Int, N_DIMS}, energy_params::EnergyParams)
    # energy of adsorption regardess of linker configuration
    energy::Float64 = -energy_params.ϵ_0 * sum(occupancy)
    for i = 1:length(occupancy)
        which_site = ind2sub(occupancy, i) # gives tuple
        # look at contribution by linkers in surrounding faces
        for which_face = 1:N_DIMS # loop over faces
            # contribution by linker to left/below (in same cage)
            energy -= energy_params.ϵ * linker_states[which_face][i] * occupancy[i]
            # contribution by linker to right/above (need to apply PBC)
            id_right =[which_site[j] for j = 1:N_DIMS]
            if id_right[which_face] == size(occupancy)[1]
                id_right[which_face] = 1
            else
                id_right[which_face] += 1
            end
            energy -= energy_params.ϵ * linker_states[which_face][id_right...] * occupancy[i]
        end
    end
    return energy
end

# The following two functions, one to rotate a given linker, the other to change the occupancy of a given adsorption site,
#    are for facilitating Markov chain moves in the Monte Carlo simulation.
"""
This function rotates linker with index `which_cage`.

Parameters:
    linker_states: see VARIABLE CONVENTIONS above
    which_face: on which of the N_DIMS faces does this ligand lie? 
    which_cage: the linker in which cage do we seek to rotate? e.g. [2,1,3]
Result:
    linker_states is modified in place
"""
function rotate_linker!(linker_states::Array{Array{Int, N_DIMS}}, which_face::Int, which_cage::Array{Int})
    if linker_states[which_face][which_cage...] == 0
        linker_states[which_face][which_cage...] = 1
    else
        linker_states[which_face][which_cage...] = 0
    end
end

"""
This function switches the occupancy of adsorption site with index `which_cage`.

Parameters:
    occupancy: see VARIABLE CONVENTIONS above
    which_cage: index of cage whose occupancy we seek to switch
Result:
    occupancy is modified in place
"""
function switch_occupancy!(occupancy::Array{Int, N_DIMS}, which_cage::Array{Int})
    if occupancy[which_cage...] == 0
        occupancy[which_cage...] = 1
    else
        occupancy[which_cage...] = 0
    end
end

# datatype used to collect statistics during the Monte Carlo simulation.
type SimStats
    # number of adsorbates
    n::Float64
    n2::Float64
    n3::Float64
    # two site correlation
    n1n2::Float64  # <n1 n2>
    # linkers rotated into the pores
    l::Float64
    l2::Float64
    # error estimates
    n_error::Float64
    l_error::Float64
    # system energies for computing heat of adsorption (includes energy of linkers)
    energy::Float64  # for <E>
    energy_times_n::Float64  # for <EN>
    qst::Float64  # heat of adsorption
    # also store energy due only to guest-host and host interaction (gh = guest-host)
    energy_gh::Float64  # for <E>
    energy_h::Float64  # for <E>
    energy_gh_times_n::Float64  # for <EN>
    energy_h_times_n::Float64  # for <EN>
    qst_gh::Float64  # heat of adsorption due to guest-host
    qst_h::Float64  # heat of adsorption due to host
    # sample count
    sample_count::Int

    function SimStats()
        sim_stats = new()
        sim_stats.n = 0.0; sim_stats.n2 = 0.0; sim_stats.n3 = 0.0;
        sim_stats.n1n2 = 0.0
        sim_stats.l = 0.0; sim_stats.l2 = 0.0;
        sim_stats.n_error = 0.0; sim_stats.l_error = 0.0;
        sim_stats.energy = 0.0; sim_stats.energy_times_n = 0.0;
        sim_stats.qst = 0.0 
        sim_stats.energy_gh = 0.0; sim_stats.energy_h = 0.0;
        sim_stats.energy_gh_times_n = 0.0; sim_stats.energy_h_times_n = 0.0
        sim_stats.qst_gh = 0.0; sim_stats.qst_h = 0.0
        sim_stats.sample_count = 0
        return sim_stats
    end
end

"""
Run grand-canonical Monte Carlo simulation of this lattice model with rotating linkers.

The state of each linker is stored in `linker_states`. The entry is `1` iff the linker is sticking into the pore, and `0` otherwise.

The occupancy state of each lattice site is stored in `occupancy`. The entry is `1` if the adsorption site is occupied by an adsorbate and `0` otherwise.

Periodic boundary conditions are applied.

Parameters:
    pressure: units in bar
    energy_params: energetic parameters (see EnergyParams type)
    n_sites: number of adsorption cages to consider in each dimension
    site_volume: volume of each adsorption site (A^3)
    samples_per_site: number of Monte Carlo samples per adsorption site (so it scales w./ # sites automatically)
    temperature: units in Kelvin
    verboseflag: print stuff?
"""
function gcmc_simulation(pressure::Float64, energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64;
            samples_per_site::Int=100, 
            verboseflag::Bool=true,
            initial_condition::String="empty")
    # thermodynamic beta to use in front of energies (ONLY, since energy is in units kJ/mol here)
    β::Float64 = 1.0 / (8.314 * temperature) * 1000  # mol / KJ to use in conjuction with energy units
    # compute total number of Monte Carlo samples
    const n_samples::Int = samples_per_site * n_sites ^ N_DIMS
    # system volume (A^3)
    const VOLUME = site_volume * n_sites ^ N_DIMS
    # initialize statitics as zeros.
    stats = SimStats()
    samples_of_n = Int[]

    # initilize occupancy and linkers.
    # linker_states[k] gives array of linkers on each face of the hypercube.
    if initial_condition == "empty"
        linker_states = [zeros(Int, [n_sites for i = 1:N_DIMS]...) for k = 1:N_DIMS]
        occupancy = zeros(Int, [n_sites for k = 1:N_DIMS]...)
        @assert(sum(occupancy) == 0)
    elseif initial_condition == "full"
        linker_states = [ones(Int, [n_sites for i = 1:N_DIMS]...) for k = 1:N_DIMS]
        occupancy = ones(Int, [n_sites for k = 1:N_DIMS]...)
        @assert(sum(occupancy) == n_sites ^ N_DIMS)
    elseif initial_condition == "random"
        linker_states = [rand(0:1, [n_sites for i = 1:N_DIMS]...) for k = 1:N_DIMS]
        occupancy = rand(0:1, [n_sites for k = 1:N_DIMS]...)
    else
        error("Initial condition must be \"empty\" or \"full\"")
    end

    # intilize adsorbate energy for heat of adsortion
    guest_energy::Float64 = system_guest_energy(linker_states, occupancy, energy_params)

    # start Monte Carlo simulation
    for i = 1:n_samples
        # decide which Markov Chain move to do.
        #   1: rotate a linker
        #   2: insert an adsorbate
        #   3: remove an adsorbate
        #  Note that prob(2) == prob(3) for detailed balance...
        which_move = rand(1:3)

        ###
        #  Propose to rotate a linker.
        ###
        if which_move == 1
            # pick a linker to propose to rotate
            which_face = rand(1:N_DIMS)
            which_cage = [rand(1:n_sites) for k = 1:N_DIMS]
            # calculate the energy of this linker
            energy_old = linker_energy(which_face, which_cage, linker_states, occupancy, energy_params)
            # rotate the linker
            rotate_linker!(linker_states, which_face, which_cage)
            # calculate the energ of this linker in its new configuration
            energy_new = linker_energy(which_face, which_cage, linker_states, occupancy, energy_params)
            # accept according to e^{-\beta dE}
            dE = energy_new - energy_old
            if rand() < exp(-β * dE)
                # accept move
                # update guest energy.
                # sign of energy change: benefit or penalty.
                # if linker is rotated into the pore now, we have a BENEFIT
                # if liker is rotated parallel to walls now, we have a PENALTY
                benefit_or_penalty = linker_states[which_face][which_cage...] == 1 ? -1.0 : 1.0

                # account for benefit/penalty for guest molecule to right (if occupancy is 1)
                guest_energy += benefit_or_penalty * energy_params.ϵ * occupancy[which_cage...]
                # account for benefit/penalty for guest molecule to left (PBC considered)
                id_left = deepcopy(which_cage)
                if which_cage[which_face] == 1
                    id_left[which_face] = n_sites
                else
                    id_left[which_face] -= 1
                end
                guest_energy += benefit_or_penalty * energy_params.ϵ * occupancy[id_left...]
            else
                # Reject move. Rotate the linker back to its previous configuration.
                rotate_linker!(linker_states, which_face, which_cage)
            end
        end # end  rotate linker move

        ###
        #  Propose to insert an adsorbate at a random lattice site
        ###
        if which_move == 2
            # pick a lattice site
            which_site = [rand(1:n_sites) for k = 1:N_DIMS]
            # do nothing if this site is already occupied
            if occupancy[which_site...] != 1
                # if we make it here, the site is empty. Propose to insert adsorbate here.
                switch_occupancy!(occupancy, which_site)
                energy = adsorbate_energy(which_site, linker_states, occupancy, energy_params)
                if rand() < pressure * VOLUME / (sum(occupancy) * k_b * temperature) * exp(-β * energy)
                    # accept move
                    guest_energy += energy
                else
                    # reject move, switch occupancy back
                    switch_occupancy!(occupancy, which_site)
                end
            end
        end

        ###
        #  Propose to delete a given adsorbate
        ###
        if which_move == 3
            # do nothing if there are no adsorbates in the system
            if sum(occupancy) != 0
                # get indices of those occupied
                idx_occupied = find(occupancy .== 1) # these are scalar indices. i.e. even for multidimensional array will be an integer.
                # choose an adsorbate. this corresponds to an entry in idx_occupied
                which_adsorbate = rand(1:length(idx_occupied))
                # get the index of the site where this adsorbate is
                which_site = ind2sub(occupancy, idx_occupied[which_adsorbate]) # gives tuple
                # calculate the energy of this adsorbate
                energy = adsorbate_energy([which_site[k] for k = 1:N_DIMS], linker_states, occupancy, energy_params)
                if rand() < sum(occupancy) * k_b * temperature / (pressure * VOLUME) * exp(β * energy)
                    # accept move, remove adsorbate
                    switch_occupancy!(occupancy, [which_site[k] for k = 1:N_DIMS])
                    guest_energy -= energy
                else
                    # reject move
                end
            end
        end # end deletion

        # if burn cycles are over, take sample every so often
        # TODO: calculate autocorrelation function to see how many samples we need before 
        #   the samples are no longer correlated.
        if (i > n_samples / 2) & (i % 10 == 0)
            stats.sample_count += 1
            # divide by sample_count later for average
            # get N and L here.
            n_here::Int = sum(occupancy)
            l_here::Int = 0
            for k = 1:N_DIMS
                l_here += sum(linker_states[k])
            end
                
            stats.n += 1.0 * n_here
            stats.n2 += (1.0 * n_here) ^ 2
            stats.n3 += (1.0 * n_here) ^ 3

            stats.l += 1.0 * l_here
            stats.l2 += 1.0 * l_here ^ 2
            
            stats.energy += guest_energy + energy_params.ϵ_l * l_here
            stats.energy_gh += guest_energy
            stats.energy_h += energy_params.ϵ_l * l_here
            stats.energy_times_n += (guest_energy + energy_params.ϵ_l * l_here) * n_here
            stats.energy_gh_times_n += guest_energy * n_here
            stats.energy_h_times_n += energy_params.ϵ_l * l_here * n_here
            
            # two site correlation function (just do in 1D)
            for j = 1:n_sites
                # <n_i n_{i+1}>
                if j == n_sites  # PBC
                    stats.n1n2 += occupancy[n_sites] * occupancy[1]
                else
                    stats.n1n2 += occupancy[j] * occupancy[j+1]
                end
 #                 # <n_i n_far> take far as middle
 #                 i_middle = i + div(n_sites, 2)
 #                 if i_middle > n_sites  # PBC
 #                     i_middle -= n_sites
 #                 end
 #                 stats.n1nfar += occupancy[i] * occupancy[i_middle]
            end

            # add to vector of samples
            push!(samples_of_n, sum(occupancy))
        end
    end # end Monte Carlo moves

    # check that we properly accounted for guest energy
    guest_energy_end::Float64 = system_guest_energy(linker_states, occupancy, energy_params)
    @test_approx_eq_eps guest_energy_end guest_energy 1e-5
 #     if abs(guest_energy_end - guest_energy) > 1e-6
 #         @printf("ERROR: guest energy = %f, end guest energy = %f\n", guest_energy, guest_energy_end)
 #     end
    # check that system_guest_energy is just sum over adsorbate energies
    sum_over_adsorbate_energy::Float64 = 0.0
    for i = 1:length(occupancy)
        which_site = ind2sub(occupancy, i)
        sum_over_adsorbate_energy += adsorbate_energy([which_site[k] for k = 1:N_DIMS], linker_states, occupancy, energy_params)
    end
    @test_approx_eq_eps sum_over_adsorbate_energy guest_energy_end 1e-5

    # turn sums into averages
    stats.n = stats.n / stats.sample_count
    stats.n2 = stats.n2 / stats.sample_count
    stats.n3 = stats.n3 / stats.sample_count
    stats.l = stats.l / stats.sample_count
    stats.l2 = stats.l2 / stats.sample_count
    stats.n1n2 = stats.n1n2 / stats.sample_count / n_sites
    stats.energy = stats.energy / stats.sample_count
    stats.energy_gh = stats.energy_gh / stats.sample_count
    stats.energy_h = stats.energy_h / stats.sample_count
    stats.energy_times_n = stats.energy_times_n / stats.sample_count
    stats.energy_gh_times_n = stats.energy_gh_times_n / stats.sample_count
    stats.energy_h_times_n = stats.energy_h_times_n / stats.sample_count
    
    # compute heat of adsorption
    stats.qst = -(stats.energy_times_n - stats.energy * stats.n) / 
        (stats.n2 - stats.n^2) + 8.314 * temperature / 1000.0
    stats.qst_gh = -(stats.energy_gh_times_n - stats.energy_gh * stats.n) / 
        (stats.n2 - stats.n^2) + 8.314 * temperature / 1000.0
    stats.qst_h = -(stats.energy_h_times_n - stats.energy_h * stats.n) / 
        (stats.n2 - stats.n^2) + 8.314 * temperature / 1000.0

    # compute confidence interval
    # Warning: this depends on how often we sample, need to look @ autocorrelation to trust this.
    stats.n_error = 1.96 * sqrt(stats.n2 - stats.n^2) / sqrt(stats.sample_count)
    stats.l_error = 1.96 * sqrt(stats.l2 - stats.l^2) / sqrt(stats.sample_count)

    if verboseflag
        @printf("T = %.1f K, P = %.2f bar\n", temperature, pressure)
        @printf("\t%d samples\n", stats.sample_count)
        print_energy_params(energy_params)
        @printf("\t%d dimensions. # lattice sites in each dim: %d, volume %f A^3 each\n", N_DIMS, n_sites, site_volume)
        @printf("\t<N> = %f\n", stats.n)
        @printf("\t<l> = %f\n", stats.l)
        @printf("\t<N> = %f +/ %f\n", stats.n, stats.n_error)
        @printf("\t<l> = %f +/ %f\n", stats.l, stats.l_error)
        @printf("\tIsosteric heat of adsorption = %f\n", stats.qst)
    end

    return occupancy, linker_states, stats, samples_of_n
end

# This will call the gcmc_simulation function at different pressures to run an adsorption isotherm.
"""
Run adsorption isotherm for a set of pressures.
Returns dataframe of properties at each pressure.

Parameters:
    pressure: array of pressures over which to compute isotherm, units in bar
    energy_params: energetic parameters (see EnergyParams type)
    n_sites: number of adsorption cages to consider
    site_volume: volume of each adsorption site (A^3)
    temperature: units in Kelvin
    samples_per_site: number of Monte Carlo samples per adsorption site (so it scales w./ # sites automatically)
Returns:
    dataframe of results
"""
function run_gcmc_isotherm(pressures::Array{Float64},
                      energy_params::EnergyParams, 
                      n_sites::Int,
                      site_volume::Float64,
                      temperature::Float64;
                      samples_per_site::Int=25000,
                      desorption_too::Bool=false)
    n_pts = length(pressures)
    if desorption_too
        n_pts += length(pressures) - 1
    end
    # store results in this dataframe
    df = DataFrame(P=zeros(Float64, n_pts),
                  n=zeros(Float64, n_pts),
                  l=zeros(Float64, n_pts), 
                  qst=zeros(Float64, n_pts), 
                  qst_gh=zeros(Float64, n_pts), 
                  qst_h=zeros(Float64, n_pts), 
                  n2=zeros(Float64, n_pts), 
                  n3=zeros(Float64, n_pts),
                  n1n2=zeros(Float64, n_pts), # two site correlation
                  desorption_branch= zeros(Bool, n_pts)
                  )

    n_true_sites = n_sites ^ N_DIMS
    
    # loop over pressures, run GCMC simulation
    n_runs = 0
    for i = 1:length(pressures)
        n_runs += 1
        occupancy, linker_states, stat = gcmc_simulation(pressures[i], energy_params, n_sites, 
                                                         site_volume, temperature, verboseflag=false, 
                                                         samples_per_site=samples_per_site)
        df[:P][i] = pressures[i]
        df[:l][i] = stat.l / n_true_sites / N_DIMS
        df[:n][i] = stat.n / n_true_sites
        df[:n2][i] = stat.n2 / n_true_sites ^ 2
        df[:n3][i] = stat.n3 / n_true_sites ^ 3
        df[:n1n2][i] = stat.n1n2
        df[:qst][i] = stat.qst
        df[:qst_gh][i] = stat.qst_gh
        df[:qst_h][i] = stat.qst_h
    end

    if desorption_too
        for j = length(pressures)-1:-1:1
            n_runs += 1
            occupancy, linker_states, stat = gcmc_simulation(pressures[j], energy_params, n_sites, 
                                                             site_volume, temperature, verboseflag=false, 
                                                             samples_per_site=samples_per_site,
                                                             initial_condition="full")

            df[:P][n_runs] = pressures[j]
            df[:l][n_runs] = stat.l / n_true_sites / N_DIMS
            df[:n][n_runs] = stat.n / n_true_sites
            df[:n2][n_runs] = stat.n2 / n_true_sites ^ 2
            df[:n3][n_runs] = stat.n3 / n_true_sites ^ 3
            df[:n1n2][n_runs] = stat.n1n2
            df[:qst][n_runs] = stat.qst
            df[:qst_gh][n_runs] = stat.qst_gh
            df[:qst_h][n_runs] = stat.qst_h
            df[:desorption_branch][n_runs] = true
        end
    end

    return df
end

"""
Plot results returned from run_isotherm.
Three panel plot.
    1. Adsorption isotherm
    2. Linker configurations
    3. Heat of adsorption

Returns Gadfly hstack'ed plot.
"""
function plot_gcmc_results(df_gcmc::DataFrame, energy_params::EnergyParams, temperature::Float64)
    xticks = collect(linspace(0.0, maximum(df_gcmc[:P]), 6))
    β = 1.0 / (8.314 * 298.0) * 1000.0  # mol / kJ

    # Panel 1: adsorption isotherm
    myplot_1 = plot(
            layer(x=df_gcmc[:P], y=df_gcmc[:n], Geom.point, Geom.line),
            Guide.xlabel("Pressure (bar)"),
            Guide.ylabel("Fractional occupancy"),
 #             Coord.Cartesian(xmax=1.0),
            Guide.xticks(ticks=xticks),
            Guide.yticks(ticks=collect(0.0:0.2:1.0)),
            Theme(background_color=colorant"white", panel_stroke=colorant"black", grid_color=colorant"Gray")
    )

    # Panel 2: Linker configurations
    myplot_2 = plot(
            layer(x=df_gcmc[:P], y=df_gcmc[:l], Geom.point, Geom.line),
            Guide.xlabel("Pressure (bar)"),
            Guide.xticks(ticks=xticks),
            Guide.ylabel("Fraction of linkers protruded"),
            Geom.hline(color=colorant"black"),
            yintercept=[1.0 / (1.0 + exp(β * energy_params.ϵ_l))],
            Theme(background_color=colorant"white", panel_stroke=colorant"black", grid_color=colorant"Gray")
    )

    # Panel 3: Qst
    myplot_3 = plot(
            layer(x=df_gcmc[:P], y=df_gcmc[:qst], Geom.point, Geom.line),
            layer(x=df_gcmc[:P], y=df_gcmc[:qst_gh], Geom.point, Geom.line, Theme(default_color=colorant"yellow")),
            Guide.xlabel("Pressure (bar)"),
            Guide.xticks(ticks=xticks),
            Guide.ylabel("Heat of adsorption (kJ/mol)"),
            yintercept=[energy_params.ϵ_0, energy_params.ϵ_0 + 2.0 * energy_params.ϵ] + 8.314 * temperature / 1000.0,
            Geom.hline(color=colorant"black"),
            Theme(background_color=colorant"white", panel_stroke=colorant"black", grid_color=colorant"Gray")
    )

    return hstack(myplot_1, myplot_2, myplot_3)
end

# NVT simulation
function NVT(N::Int, energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64;
            samples_per_site::Int=100, 
            verboseflag::Bool=true)
    # thermodynamic beta to use in front of energies (ONLY, since energy is in units kJ/mol here)
    β::Float64 = 1.0 / (8.314 * temperature) * 1000  # mol / KJ to use in conjuction with energy units
    # compute total number of Monte Carlo samples
    const n_samples::Int = samples_per_site * n_sites ^ N_DIMS
    const VOLUME = site_volume * n_sites ^ N_DIMS
    @assert(N > 0)
    # initialize statitics as zeros.
    stats = SimStats()

    # initilize occupancy and linkers.
    # linker_states[k] gives array of linkers on each face of the hypercube.
    linker_states = [rand(0:1, [n_sites for i = 1:N_DIMS]...) for k = 1:N_DIMS]
    occupancy = zeros(Int, [n_sites for k = 1:N_DIMS]...)
    idx_occupied = rand(1:n_sites^2)
    while sum(occupancy) != N
        occupancy[rand(1:n_sites, N_DIMS)...] = 1
    end
    @assert(sum(occupancy) == N)

    # intilize adsorbate energy for heat of adsortion
    guest_energy::Float64 = system_guest_energy(linker_states, occupancy, energy_params)

    # start Monte Carlo simulation
    for i = 1:n_samples
        # decide which Markov Chain move to do.
        #   1: rotate a linker
        #   2: reinsert an adsorbate
        #  Note that prob(2) == prob(3) for detailed balance...
        which_move = rand(1:2)

        ###
        #  Propose to rotate a linker.
        ###
        if which_move == 1
            # pick a linker to propose to rotate
            which_face = rand(1:N_DIMS)
            which_cage = [rand(1:n_sites) for k = 1:N_DIMS]
            # calculate the energy of this linker
            energy_old = linker_energy(which_face, which_cage, linker_states, occupancy, energy_params)
            # rotate the linker
            rotate_linker!(linker_states, which_face, which_cage)
            # calculate the energ of this linker in its new configuration
            energy_new = linker_energy(which_face, which_cage, linker_states, occupancy, energy_params)
            # accept according to e^{-\beta dE}
            dE = energy_new - energy_old
            if rand() < exp(-β * dE)
                # accept move
                # update guest energy.
                # sign of energy change: benefit or penalty.
                # if linker is rotated into the pore now, we have a BENEFIT
                # if liker is rotated parallel to walls now, we have a PENALTY
                benefit_or_penalty = linker_states[which_face][which_cage...] == 1 ? -1.0 : 1.0

                # account for benefit/penalty for guest molecule to right (if occupancy is 1)
                guest_energy += benefit_or_penalty * energy_params.ϵ * occupancy[which_cage...]
                # account for benefit/penalty for guest molecule to left (PBC considered)
                id_left = deepcopy(which_cage)
                if which_cage[which_face] == 1
                    id_left[which_face] = n_sites
                else
                    id_left[which_face] -= 1
                end
                guest_energy += benefit_or_penalty * energy_params.ϵ * occupancy[id_left...]
            else
                # Reject move. Rotate the linker back to its previous configuration.
                rotate_linker!(linker_states, which_face, which_cage)
            end
        end # end  rotate linker move

        ###
        #  Propose to reinsert an adsorbate at a new location
        ###
        if which_move == 2
            # get indices of those occupied
            idx_occupied = find(occupancy .== 1) # these are scalar indices. i.e. even for multidimensional array will be an integer.
            # choose an adsorbate. this corresponds to an entry in idx_occupied
            which_adsorbate = rand(1:length(idx_occupied))
            # get the index of the site where this adsorbate is
            which_site_tuple = ind2sub(occupancy, idx_occupied[which_adsorbate]) # gives tuple
            which_site = [which_site_tuple[k] for k = 1:N_DIMS]
            # calculate the energy of this adsorbate
            energy_old = adsorbate_energy(which_site, linker_states, occupancy, energy_params)
            # pick a new site at random
            new_site = rand(1:n_sites, N_DIMS)
            @assert(occupancy[which_site...] == 1)
            # if new site is unoccupied
            if occupancy[new_site...] == 0
                # move the adsorbate to this site
                occupancy[which_site...] = 0
                occupancy[new_site...] = 1
                energy_new = adsorbate_energy(new_site, linker_states, occupancy, energy_params)
                if rand() < exp(- β * (energy_new - energy_old))
                    # accept move!
                    guest_energy += energy_new - energy_old
                else
                    # reject move, move back adsorbate
                    occupancy[which_site...] = 1
                    occupancy[new_site...] = 0
                end
            end
        end # end deletion

        # if burn cycles are over, take sample every so often
        # TODO: calculate autocorrelation function to see how many samples we need before 
        #   the samples are no longer correlated.
        if (i > n_samples / 2) & (i % 10 == 0)
            stats.sample_count += 1
            # divide by sample_count later for average
            # linker configurations
            for k = 1:N_DIMS
                stats.l += 1.0 * sum(linker_states[k])
            end
        end
        if sum(occupancy) != N
            @printf("N = %d, sum(occupancy) = %d\n", N, sum(occupancy))
        end
    end # end Monte Carlo moves

    # check that we properly accounted for guest energy
    guest_energy_end::Float64 = system_guest_energy(linker_states, occupancy, energy_params)
    @test_approx_eq_eps guest_energy_end guest_energy 1e-5
    # check that system_guest_energy is just sum over adsorbate energies
    sum_over_adsorbate_energy::Float64 = 0.0
    for i = 1:length(occupancy)
        which_site = ind2sub(occupancy, i)
        sum_over_adsorbate_energy += adsorbate_energy([which_site[k] for k = 1:N_DIMS], linker_states, occupancy, energy_params)
    end
    @test_approx_eq_eps sum_over_adsorbate_energy guest_energy_end 1e-5

    # turn sums into averages
    stats.n = stats.n / stats.sample_count
    stats.l = stats.l / stats.sample_count

    if verboseflag
        @printf("T = %.1f K\n", temperature)
        @printf("\t%d samples\n", stats.sample_count)
        print_energy_params(energy_params)
        @printf("\t%d dimensions. # lattice sites in each dim: %d, volume %f A^3 each\n", N_DIMS, n_sites, site_volume)
        @printf("\t<N> = %f\n", stats.n)
        @printf("\t<l> = %f\n", stats.l)
    end

    return occupancy, linker_states, stats
end

