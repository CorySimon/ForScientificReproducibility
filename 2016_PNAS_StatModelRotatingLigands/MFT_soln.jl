# Mean field solution
include("utils.jl")
using Colors
using DataFrames
using Gadfly
using NLsolve  # nlsolve()
using NLopt # Opt()

"""
Self-consistency equation for ℓ;
Given 𝑛 / 𝑀, return ℓ / (𝑀 d).

Parameters:
    n: 𝑛 / 𝑀
    energy_params: energetic parameters of model (see EnergyParams in utils.jl)
    temperature: units K
"""
function self_consistency_ℓ(n::Array{Float64}, energy_params::EnergyParams, temperature::Float64)
    β = 1.0 / (8.314 * temperature) * 1000  # mol / KJ to use in conjuction with energy units
    return 1.0 ./ (1.0 + exp(β * (energy_params.ϵ_l - 2.0 * energy_params.ϵ * n)))
end

"""
Self-consistency equation for 𝑛;
Given ℓ / 𝑀 and pressure, return 𝑛 / 𝑀.

Parameters:
    l: ℓ / (𝑀 d)
    pressure: units: bar
    energy_params: energetic parameters of model (see EnergyParams in utils.jl)
    temperature: units K

"""
function self_consistency_n(l::Array{Float64}, pressure::Float64, energy_params::EnergyParams, site_volume::Float64, temperature::Float64)
    β::Float64 = 1.0 / (8.314 * temperature) * 1000  # mol / KJ to use in conjuction with energy units
    K = exp(β * (energy_params.ϵ_0 + 2.0 * N_DIMS * l * energy_params.ϵ)) / (k_b * temperature) * site_volume
    return K * pressure ./ (1.0 + K * pressure)
end

"""
Plot graphical solution of MFT model.
i.e. the self-consistency equations of n and l.

Parameters:
    pressures: for each pressure in this array (units: bar), we plot a self-consistency eqn for n.
    energy_params: energetic parameters of model (see EnergyParams in utils.jl)
    site_volume: free volume of cage (A^3)
    temperature: units K

Returns: Gadfly plot
"""
function plot_graphical_MFT_soln(pressures::Array{Float64}, energy_params::EnergyParams,
                                 site_volume::Float64, temperature::Float64)
    # ℓ self-consistency eqn.
    n_plot = collect(linspace(0, 1, 100))
    l = self_consistency_ℓ(n_plot, energy_params, temperature)

    # 𝑛 self-consitency eqn. one for each pressure
    l_plot = collect(linspace(0, 1, 100))
    n = zeros(Float64, length(l_plot), length(pressures))  # store results here
    for i = 1:length(pressures)
        n[:, i] = self_consistency_n(l_plot, pressures[i], energy_params, site_volume, temperature)
    end
    
    # colormap for depicting value of pressure in each n-curve.
    colors = colormap("Blues", length(pressures_mft))

    β = 1.0 / (8.314 * temperature) * 1000  # mol / KJ to use in conjuction with energy units
    myplot = plot(
        # ℓ self-consistency eqn.
        layer(x=n_plot, y=l, Geom.line, Theme(default_color=colorant"red", line_width=.7mm)),
        # 𝑛 self-consitency eqn. one for each pressure
        [layer(x=n[:, i], y=l_plot, Geom.line, Theme(default_color=colors[i], line_width=.7mm)) for i = 1:length(pressures)]...,
        Guide.manual_color_key("𝑃 (bar)",
            [@sprintf("%.3f", pressures[i]) for i = 1:length(pressures)], 
            colors
        ),
        Guide.xlabel("𝑛/𝑀"),
        Guide.ylabel("ℓ/(𝑀d)"),
        Guide.xticks(ticks=collect(0.0:0.2:1.0)),
        Guide.yticks(ticks=collect(0.0:0.2:1.0)),
 #         Guide.title("Graphical solution"),
        Theme(panel_fill=RGB(214./255.0, 214./255.0,214./255.0), 
            panel_opacity=0.8,
            grid_color=colorant"Gray",
            panel_stroke=colorant"black",
            background_color=colorant"white",
            line_width=.7mm, major_label_font_size=15pt, minor_label_font_size=14pt,
            key_label_font_size=13pt, key_title_font_size=16pt,
            minor_label_color=colorant"black", major_label_color=colorant"black", 
            key_title_color=colorant"black", key_label_color=colorant"black"
        )
    )
    return myplot
end

"""
Get MFT solution.

Parameters:
    pressure: units in bar
    energy_params: energetic parameters (see EnergyParams type)
    n_sites: number of adsorption cages to consider
    site_volume: volume of each adsorption site (A^3)
    temperature: units in Kelvin
    nl_guess: [n/M, l/(Md)] guess as starting point for nonlinear solver

Returns: [n/M, l/(Md)] in MFT model
"""
function mft_soln(pressure::Float64, energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64;
                    nl_guess::Array{Float64}=[.2, .2])
    β = 1.0 / (8.314 * temperature) * 1000  # mol / KJ to use in conjuction with energy units

    function l_eqn(nl::Array{Float64})
        """
        nl[1] = n / M
        nl[2] = l / (Md)

        MFT eqn for l. This should be zero
        """
        return nl[2] - 1.0 / (1.0 + exp(β * (energy_params.ϵ_l - 2.0 * nl[1] * energy_params.ϵ)))
    end

    function n_eqn(nl::Array{Float64})
        """
        nl[1] = n / M
        nl[2] = l / (Md)

        MFT eqn for n. This should be zero
        """
        K = exp(β * (energy_params.ϵ_0 + 2.0 * nl[2] * N_DIMS * energy_params.ϵ)) / (k_b * temperature) * site_volume
        return nl[1] - K * pressure / (1.0 + K * pressure)
    end
    
    # Goal is to make both components of consistency_eqns zero
    # by changing nl = [n / M, l / M]
    function f!(nl, consistency_eqns)
        consistency_eqns[1] = l_eqn(nl)
        consistency_eqns[2] = n_eqn(nl)
    end
    
    # Solve nonlinear system of two self-consistency eqns.
    nl0 = deepcopy(nl_guess);
    res = nlsolve(f!, nl0);

    @assert(res.f_converged)

    return res.zero
end

"""
Get MFT adsorption isotherm.
Finds multiple solutions by changing starting guess, to capture hysteresis.

Parameters:
    pressures: array of pressures, units in bar
    energy_params: energetic parameters (see EnergyParams type)
    n_sites: number of adsorption cages to consider
    site_volume: volume of each adsorption site (A^3)
    temperature: units in Kelvin

Returns: DataFrame for adsorption and desorption branch
    df["adsorption"]
    df["desorption"]
"""
function mft_isotherm(pressures::Array{Float64}, energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64)
    # store result here.
    df = Dict{AbstractString, DataFrame}()
    df["adsorption"] = DataFrame(P=Float64[], n=Float64[], l=Float64[])
    df["desorption"] = DataFrame(P=Float64[], n=Float64[], l=Float64[])

    for i = 1:length(pressures)
        try
            nl_guess = [.01, .01]
            nl = mft_soln(pressures[i], energy_params, n_sites, site_volume, temperature, nl_guess=nl_guess)
            push!(df["adsorption"], [pressures[i], nl[1], nl[2]])
        catch
            # no soln with this guess
        end

        # to avoid zero solution that connects desorption branch to zero awkwardly in plot
 #         if pressures[i] < 0.1
        if pressures[i] < 0.004
            continue
        end

        try
            nl_guess = [.99, .99]
            nl = mft_soln(pressures[i], energy_params, n_sites, site_volume, temperature, nl_guess=nl_guess)
            push!(df["desorption"], [pressures[i], nl[1], nl[2]])
        catch
            # no soln with this guess
        end
    end
    return df
end

"""
Find the jump in the MFT solution.
Do this by enforcing self-consistency eqns hold. To constrain the pressure at which the jump occurs,
    also enforce that the curves are tangent.

Parameters:
    pressures: array of pressures, units in bar
    energy_params: energetic parameters (see EnergyParams type)
    n_sites: number of adsorption cages to consider
    site_volume: volume of each adsorption site (A^3)
    temperature: units in Kelvin
    nlp_guess: [n/M, l/M, p] @ infleciton as guess as starting point for nonlinear solver
"""
function find_mft_jump(energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64, nlp_guess::Array{Float64})
    β = 1.0 / (8.314 * temperature) * 1000  # mol / KJ to use in conjuction with energy units
    ϵ = energy_params.ϵ
    ϵ_l = energy_params.ϵ_l
    ϵ_0 = energy_params.ϵ_0

    function l_eqn(nlp::Array{Float64})
        """
        nlp[1] = n / M
        nlp[2] = l / (Md)
        nlp[3] = p

        MFT eqn for l. This should be zero
        """
        return nlp[2] - 1.0 / (1.0 + exp(β * (ϵ_l - 2.0 * nlp[1] * ϵ)))
        end

    function n_eqn(nlp::Array{Float64})
        """
        nlp[1] = n / M
        nlp[2] = l / (Md)
        nlp[3] = p

        MFT eqn for n. This should be zero
        """
        K = exp(β * (ϵ_0 + 2.0 * N_DIMS * nlp[2] * ϵ)) / (k_b * temperature) * site_volume
        return nlp[1] - K * nlp[3] / (1.0 + K * nlp[3])
    end
    
    function equal_tangents(nlp::Array{Float64})
        """
        nlp[1] = n / M
        nlp[2] = l / (Md)
        nlp[3] = p

        eqn such that self-consistency eqns are tangent in the l-n plane
        """
        return 1.0 - nlp[1] * (1 - nlp[1]) * 4 * (β * ϵ) ^ 2 * nlp[2] ^ 2 * exp(β * (ϵ_l - 2 * ϵ * nlp[1])) * N_DIMS
    end

    """
    Minimize this, sum of square eqns that must be zero
    """
    function my_objective_function(nlp::Vector, grad::Vector)
        if length(grad) > 0
            error("Not giving you the gradient.")
        end
        return equal_tangents(nlp) ^ 2 + n_eqn(nlp) ^ 2 + l_eqn(nlp) ^2
    end

    opt = Opt(:LN_COBYLA, 3)
    lower_bounds!(opt, [0.0, 0.0, 0.0])
    upper_bounds!(opt, [1.0, 1.0, Inf])
    xtol_rel!(opt, 1e-4)

    min_objective!(opt, my_objective_function)

    (minf,minx,ret) = NLopt.optimize(opt, nlp_guess)
    return minx  # [n / M, l / M, p] @ jump
end

"""
Return Grand potential of the MFT model. (kJ/mol)

Parameters:
    energy_params: see type, energetic parameters in the model.
    n_sites: number of cages
    site_volume: in A3
    temperature: in K
    pressure: in bar
    n: number of occupied cages (NOT fraction)
    l: number of linkers that are rotated (NOT fraction)
"""
function Ω_mft(energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64, pressure::Float64, n::Float64, l::Float64)
    RT = 8.314 * temperature / 1000.0 # kJ/mol
    # energies (kJ)
    intrahost_energy = energy_params.ϵ_l * l
    guest_energy = - n * (energy_params.ϵ_0 + 2 * energy_params.ϵ * l / n_sites)
    # chemical potential term (kJ) note Λ term cancels from translational entropy
    # kb term in units bar-A3 / K
    # pressure in bar
    # site volume in A3. works out.
    μn = RT * n * log(site_volume * pressure / (k_b * temperature))
    # entropy term (kJ) (applied Stirling) neglect Λ term b/c it canceled from chemical potential. also neglect n_sites log(n_sites) term b/c just a shift
    # contribution from n
    TS_n = -RT * ((n_sites - n) * log(n_sites - n) + n * log(n))
    if (n == 0) | (n == n_sites)
        TS_n = -RT * n_sites * log(n_sites)
    end
    # contribution from l
    TS_l = -RT * ((n_sites * N_DIMS - l) * log(n_sites * N_DIMS - l) + l * log(l))
    if (l == 0) | (l == n_sites * N_DIMS)
        TS_l = -RT * N_DIMS * n_sites * log(n_sites * N_DIMS)
    end
    return intrahost_energy + guest_energy - TS_n - TS_l - μn
end
