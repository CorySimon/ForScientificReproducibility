using Roots
using DataFrames
include("utils.jl")

"""
Returns transfer matrix of model.

Parameters:
    pressure: units in bar
    energy_params: energetic parameters (see EnergyParams type)
    n_sites: number of adsorption cages to consider
    site_volume: volume of each adsorption site (A^3)
    temperature: units in Kelvin
"""
function get_transfer_matrix(pressure::Float64, energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64)
    β = 1.0 / (8.314 * temperature) * 1000  # mol / KJ to use in conjuction with energy units
    # unpack energetic parameters for ease
    ϵ = energy_params.ϵ
    ϵ_0 = energy_params.ϵ_0
    ϵ_l = energy_params.ϵ_l

    # construct transfer matrix
    P = zeros(Float64, 2, 2)
    P[1, 1] = 1.0 + exp(-β * ϵ_l)
    P[2, 2] = site_volume * pressure / (k_b * temperature) * exp(β * ϵ_0) * (1.0 + exp(β * (2 * ϵ - ϵ_l)))
    P[1, 2] = (1 + exp(β * (ϵ - ϵ_l))) * sqrt(site_volume * pressure / (k_b * temperature) * exp(β * ϵ_0))
    P[2, 1] = (1 + exp(β * (ϵ - ϵ_l))) * sqrt(site_volume * pressure / (k_b * temperature) * exp(β * ϵ_0))

    return P
end

"""
Compute exact solution of the model

Parameters:
    pressure: units in bar
    energy_params: energetic parameters (see EnergyParams type)
    n_sites: number of adsorption cages to consider
    site_volume: volume of each adsorption site (A^3)
    temperature: units in Kelvin

Returns: dictionary of model solutions
"""
function exact_soln(pressure::Float64, energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64)
    # Thermodynamic beta. Use units mol/kJ in conjunction with kJ/mol energy units.
    β = 1.0 / (8.314 * temperature) * 1000
    # unpack energetic parameters for conveninece
    ϵ = energy_params.ϵ
    ϵ_0 = energy_params.ϵ_0
    ϵ_l = energy_params.ϵ_l

    # get transfer matrix, its trace, and its determinant
    P = get_transfer_matrix(pressure, energy_params, n_sites, site_volume, temperature)
    TrP = trace(P)
    detP = det(P)

    # compute value of dominant eigenvalue
    stuff_in_sqrt = TrP ^ 2 - 4.0 * detP
    eigenvalue_large = (TrP + sqrt(stuff_in_sqrt)) / 2

    # Derivatives of trace(P)
    function derivative_trace_P(wrt::AbstractString)
        if wrt == "βμ"
            return TrP - (1.0 + exp(-β * ϵ_l))
        elseif wrt == "β"
            return site_volume * pressure / (k_b * temperature) * exp(β * ϵ_0) * 
                    (ϵ_0 + (ϵ_0 + 2 * ϵ - ϵ_l) * exp(β * (2 * ϵ - ϵ_l))) - ϵ_l * exp(-β * ϵ_l)
        elseif wrt == "β|βϵ_l" # i.e. with βϵ_l held constant for only adsorbate-host interactions
            return site_volume * pressure / (k_b * temperature) * exp(β * ϵ_0) * 
                    (ϵ_0 + (ϵ_0 + 2 * ϵ ) * exp(β * (2 * ϵ - ϵ_l)))
        elseif wrt == "βϵ_l"
            return -exp(-β * ϵ_l) * (1 + site_volume * pressure / (k_b * temperature) * exp(β * (ϵ_0 + 2 * ϵ)))
        elseif wrt == "βμβμ"
            return TrP - (1.0 + exp(-β * ϵ_l))
        elseif wrt == "ββμ"
            return derivative_trace_P("β") + ϵ_l * exp(-β * ϵ_l)
        elseif wrt == "β|βϵ_lβμ"
            return derivative_trace_P("β|βϵ_l") 
        elseif wrt == "βμβμβμ"
            return TrP - (1.0 + exp(-β * ϵ_l))
        elseif wrt == "βϵ_0"
            return TrP - (1 + exp(-β * ϵ_l))
        elseif wrt == "βϵ"
            return 2 * site_volume * pressure / (k_b * temperature) * exp(β * (ϵ_0 + 2 * ϵ - ϵ_l))
        elseif wrt == "βϵ_l"
            return -site_volume * pressure / (k_b * temperature) * exp(β * (ϵ_0 + 2 * ϵ - ϵ_l)) - exp(-β * ϵ_l)
        elseif wrt == "βϵ_0βμ"
            return derivative_trace_P("βϵ_0")
        elseif wrt == "βϵβμ"
            return derivative_trace_P("βϵ")
        elseif wrt == "βϵ_lβμ"
            return derivative_trace_P("βϵ_l") + exp(-β * ϵ_l)
        else
            error(@sprintf("d Tr(P) / d (%s) not implemented", wrt))
        end
    end
    
    # Derivatives of determinant
    function derivative_det_P(wrt::AbstractString)
        if wrt == "βμ"
            return detP 
        elseif wrt == "βϵ_l"
            return - detP
        elseif wrt == "β"
            return site_volume * pressure / (k_b * temperature) * exp(β * (ϵ_0 - ϵ_l)) * (1 - exp(β * ϵ)) * 
                    ((ϵ_0 - ϵ_l) * (1 - exp(β * ϵ)) - 2 * ϵ * exp(β * ϵ))
        elseif wrt == "β|βϵ_l" # i.e. with βϵ_l held constant for only adsorbate-host interactions
            return site_volume * pressure / (k_b * temperature) * exp(β * (ϵ_0 - ϵ_l)) * (1 - exp(β * ϵ)) * 
                    (ϵ_0 * (1 - exp(β * ϵ)) - 2 * ϵ * exp(β * ϵ))
        elseif wrt == "βμβμ"
            return detP
        elseif wrt == "ββμ"
            return derivative_det_P("β")
        elseif wrt == "β|βϵ_lβμ" # i.e. with βϵ_l held constant for only adsorbate-host interactions
            return derivative_det_P("β|βϵ_l")
        elseif wrt == "βμβμβμ"
            return detP
        elseif wrt == "βϵ_0"
            return detP
        elseif wrt == "βϵ"
            return - 2 * site_volume * pressure / (k_b * temperature) * exp(β * (ϵ_0 + ϵ - ϵ_l)) * (1 - exp(β * ϵ))
        elseif wrt == "βϵ_l"
            return - detP
        elseif wrt == "βϵ_0βμ"
            return derivative_det_P("βϵ_0")
        elseif wrt == "βϵβμ"
            return derivative_det_P("βϵ")
        elseif wrt == "βϵ_lβμ"
            return derivative_det_P("βϵ_l")
        else
            error(@sprintf("d det(P) / d (%s) not implemented", wrt))
        end
    end

    # first derivative of dominant eigenvalue
    function derivative_lambda(wrt::AbstractString)
        return 1 / 2 * (derivative_trace_P(wrt) + 1.0 / (2.0 * sqrt(stuff_in_sqrt)) * 
                (2 * TrP * derivative_trace_P(wrt) - 4 * derivative_det_P(wrt)))
    end

    # second derivative of dominant eigenvalue (can be mixed)
    function derivative_lambda2(wrt_y::AbstractString, wrt_x::AbstractString)
        return 1 / 2 * (
            derivative_trace_P(wrt_y * wrt_x) 
            - 1 / (4 * (stuff_in_sqrt)^(3/2)) * 
            (2 * TrP * derivative_trace_P(wrt_y) - 4 * derivative_det_P(wrt_y)) *
            (2 * TrP * derivative_trace_P(wrt_x) - 4 * derivative_det_P(wrt_x))
            + 1 / (2 * sqrt(stuff_in_sqrt)) * (2 * derivative_trace_P(wrt_y) * derivative_trace_P(wrt_x) + 
                2 * TrP * derivative_trace_P(wrt_y * wrt_x) - 4 * derivative_det_P(wrt_y * wrt_x))
            )
    end

    # third derivative of dominant eigenvalue (cant be mixed)
    function derivative_lambda3(wrt::AbstractString)
        line_1 = derivative_trace_P(wrt * wrt * wrt) +
            3 / (8 * stuff_in_sqrt ^ (5/2)) * (2 * TrP * derivative_trace_P(wrt) - 4 * derivative_det_P(wrt)) ^ 3
        line_2 = - 3 / (4 * stuff_in_sqrt^(3/2)) * (2 * TrP * derivative_trace_P(wrt) - 4 * derivative_det_P(wrt)) *
            (2 * derivative_trace_P(wrt) ^ 2 + 2 * TrP * derivative_trace_P(wrt * wrt) - 4 * derivative_det_P(wrt * wrt))
        line_3 = 1 / (2 * sqrt(stuff_in_sqrt)) *
            (6 * derivative_trace_P(wrt) * derivative_trace_P(wrt * wrt) + 2 * TrP * derivative_trace_P(wrt * wrt * wrt) - 
            4 * derivative_det_P(wrt * wrt * wrt))
        return (line_1 + line_2 + line_3) / 2
    end
   
    ###
    #  Expected values
    ###
    # <n>
    expected_n = n_sites * derivative_lambda("βμ") / eigenvalue_large
    # <n^2>
    expected_n2 = n_sites * derivative_lambda2("βμ", "βμ") / eigenvalue_large + 
                  n_sites * (n_sites - 1) * (derivative_lambda("βμ") / eigenvalue_large) ^ 2
    # <n^3>
    expected_n3 = n_sites * derivative_lambda3("βμ") / eigenvalue_large +
                  n_sites * (n_sites-1) * (n_sites -2 ) * derivative_lambda("βμ") ^3 / eigenvalue_large ^3 + 
                  3 * (n_sites-1) * n_sites * derivative_lambda("βμ") * derivative_lambda2("βμ", "βμ") / eigenvalue_large ^2
    # var(n)
    var_n = expected_n2 - expected_n ^ 2
    # <l/M>
    expected_l = -derivative_lambda("βϵ_l") / eigenvalue_large
    # <E> 
    expected_e = - n_sites * derivative_lambda("β") / eigenvalue_large 
    # <E_h>
    expected_eh = n_sites * expected_l * ϵ_l
    # <E_guest-host> 
    expected_egh = - n_sites * derivative_lambda("β|βϵ_l") / eigenvalue_large 
    # <E n>
    expected_en = - n_sites * derivative_lambda2("β", "βμ") / eigenvalue_large -
                  n_sites * (n_sites - 1) * derivative_lambda("β") * derivative_lambda("βμ") / eigenvalue_large ^ 2
    # <E_guest-host n>
    expected_eghn = - n_sites * derivative_lambda2("β|βϵ_l", "βμ") / eigenvalue_large -
                  n_sites * (n_sites - 1) * derivative_lambda("β|βϵ_l") * derivative_lambda("βμ") / eigenvalue_large ^ 2
    # <E_host n>
    expected_ehn = - n_sites * derivative_lambda2("βϵ_l", "βμ") / eigenvalue_large -
                  n_sites * (n_sites - 1) * derivative_lambda("βϵ_l") * derivative_lambda("βμ") / eigenvalue_large ^ 2
    expected_ehn = expected_ehn * ϵ_l
    # Qst, isosteric heat of adsorption
    qst = -(expected_en - expected_e * expected_n) / var_n + 8.314 * temperature / 1000.0
    # Qst, isosteric heat of adsorption, due only to host-adsorbate
    qst_gh = -(expected_eghn - expected_egh * expected_n) / var_n + 8.314 * temperature / 1000.0
    # Qst, isosteric heat of adsorption, due only to host
    qst_h = -(expected_ehn - expected_eh * expected_n) / var_n + 8.314 * temperature / 1000.0

    ###
    #  For computing the inflection point. See May 6 piece of paper
    ###
    # d<n> / d(βμ)
    dn_dbmu = derivative_lambda2("βμ", "βμ")  / eigenvalue_large - (derivative_lambda("βμ") / eigenvalue_large) ^ 2
    # d2<n> / d(βμ)2
    d2n_dbmu2 = - derivative_lambda("βμ") * derivative_lambda2("βμ", "βμ") / eigenvalue_large ^ 2 + 
        derivative_lambda3("βμ") / eigenvalue_large + 
        2 * derivative_lambda("βμ") ^ 3 / eigenvalue_large ^ 3 -
        2 * derivative_lambda("βμ") * derivative_lambda2("βμ", "βμ") / eigenvalue_large ^ 2
    # d2<n> / dP2
    d2n_dp2 = (d2n_dbmu2 - dn_dbmu) / pressure ^ 2
    
    ###
    #  Compute two-site occupancy covariance.
    ###
    # find eigenvalues and corresponding eigenvectors (i.e. diagonalize it)
    λ, U = eig(P)

    # build grand-canonical partition function
    Ξ = λ[1] ^ n_sites + λ[2] ^ n_sites

    # constructe diagonal matrix of eigenvalues for diagonalization
    Λ = zeros(Float64, 2, 2)
    Λ[1, 1] = λ[1]
    Λ[2, 2] = λ[2]

    # construct Phi matrix
    Φ = zeros(Float64, 2, 2)
    Φ[2, 2] = 1.0

    # compute <n_1 n_2>
    n1n2 = trace(transpose(U) * Φ * U * Λ * transpose(U) * Φ * U * Λ ^ (n_sites - 1)) / Ξ
    # compute <n_1 n_far>, where far is cage in middle due to PBC
    i_middle = div(n_sites, 2)
    n1nfar = trace(transpose(U) * Φ * U * Λ^ (i_middle) * transpose(U) * Φ * U * Λ ^ (n_sites - i_middle)) / Ξ

    ###
    #  For maximizing the working capacity, d <n> / d ϵ = 0.
    #     => need d / dϵ ( d log Ξ / d (βμ) ) = 0
    ###
    dn_d = Dict{AbstractString, Float64}()
    for param in ["βϵ_0", "βϵ", "βϵ_l"]
        dn_d[param] = -derivative_lambda(param) * derivative_lambda("βμ") / eigenvalue_large ^ 2 + 
                        derivative_lambda2(param, "βμ") / eigenvalue_large
    end


    # normalize by number of sites
    return Dict("n" => expected_n / n_sites, 
                "n2" => expected_n2 / n_sites ^ 2,
                "n3" => expected_n3 / n_sites ^ 3,
                "n1n2" => n1n2,
                "n1nfar" => n1nfar,
                "l" => expected_l, 
                "energy" => expected_e,
                "energy_h" => expected_eh,
                "energy_gh" => expected_egh,
                "qst" => qst,
                "qst_gh" => qst_gh,
                "qst_h" => qst_h,
                "dn_dp" => dn_dbmu / pressure,
                "d2n_dp2" => d2n_dp2,
                "dn_dβϵ_0" => dn_d["βϵ_0"],
                "dn_dβϵ" => dn_d["βϵ"],
                "dn_dβϵ_l" => dn_d["βϵ_l"]
    )
end

"""
Compute Qst using numerical differentiation.
see http://pubs.acs.org/doi/pdf/10.1021/la00060a035

qst = RT - dU / dN

use centered difference for the derivative with spacing dp

returns qst, contribution by guest-host, contribution by host
"""
function qst_by_differentiation(pressure::Float64, energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64; dp::Float64=1e-6)
    high_soln = exact_soln(pressure + dp/2, energy_params, n_sites, site_volume, temperature)
    low_soln = exact_soln(pressure - dp/2, energy_params, n_sites, site_volume, temperature)
    dU_dn = (high_soln["energy"] - low_soln["energy"]) / n_sites / (high_soln["n"] - low_soln["n"])
    dUgh_dn = (high_soln["energy_gh"] - low_soln["energy_gh"]) / n_sites / (high_soln["n"] - low_soln["n"])
    dUh_dn = (high_soln["energy_h"] - low_soln["energy_h"]) / n_sites / (high_soln["n"] - low_soln["n"])
    RT = 8.314 * temperature / 1000.0
    return RT - dU_dn, - dUgh_dn, - dUh_dn
end

"""
Compute exact solution as a function of pressure.

Parameters:
    pressures: array of pressures, units in bar
    energy_params: energetic parameters (see EnergyParams type)
    n_sites: number of adsorption cages to consider
    site_volume: volume of each adsorption site (A^3)
    temperature: units in Kelvin

Returns: DataFrame with result.
"""
function get_exact_isotherm(pressures::Array{Float64}, energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64)
    # store result in this dataframe.
    df = DataFrame(P=pressures,
                   n=similar(pressures),
                   l=similar(pressures), 
                   energy=similar(pressures),
                   qst=similar(pressures), 
                   qst_gh=similar(pressures), 
                   qst_h=similar(pressures), 
                   n2=similar(pressures), 
                   n3=similar(pressures),
                   n1n2=similar(pressures), # two site correlation
                   n1nfar = similar(pressures),
                   dn_dp=similar(pressures),
                   d2n_dp2=similar(pressures)
    )

    # Loop over pressures, compute exact soln, store quantities in the dataframe.
    for i = 1:length(pressures)
        soln = exact_soln(pressures[i], energy_params, n_sites, site_volume, temperature)
        df[:n][i] = soln["n"]
        df[:n2][i] = soln["n2"]
        df[:n3][i] = soln["n3"]
        df[:n1n2][i] = soln["n1n2"]
        df[:n1nfar][i] = soln["n1nfar"]
        df[:l][i] = soln["l"]
        df[:energy][i] = soln["energy"]
        df[:qst][i] = soln["qst"]
        df[:qst_gh][i] = soln["qst_gh"]
        df[:qst_h][i] = soln["qst_h"]
        df[:dn_dp][i] = soln["dn_dp"]
        df[:d2n_dp2][i] = soln["d2n_dp2"]
    end
    return df
end

"""
Find inflection point by computing the maximum of the first derivative.

Parameters:
    energy_params: energetic parameters (see EnergyParams type)
    n_sites: number of adsorption cages to consider
    site_volume: volume of each adsorption site (A^3)
    temperature: units in Kelvin
Returns: pressure at inflection (bar), n at inflection
"""
function compute_inflection_pt(energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64)
    # Write function that returns d^2n/ dp^2 as a function of pressure.
    # We seek this to be zero for an inflection point.
    function d2n_dp2(pressure)
        soln = exact_soln(pressure, energy_params, n_sites, site_volume, temperature)
        return soln["d2n_dp2"]
    end
    
    # Use Roots.jl to find zero of d^n/dp^2.
    inflection_pressure::Float64 = NaN
    try
        inflection_pressure = fzero(d2n_dp2, 1e-8, 3.0)
    catch
        # if root finding fails, return NaN
        return NaN, NaN
    end
    # find value of n at inflection.
    soln = exact_soln(inflection_pressure, energy_params, n_sites, site_volume, temperature)
    inflection_n = soln["n"]

    return inflection_pressure, inflection_n
end

"""
Compute working capacity in model.

Parameters:
    P_H: high pressure in pressure swing adsorption process (bar)
    P_L: high pressure in pressure swing adsorption process (bar)
    energy_params: energetic parameters (see EnergyParams type)
    n_sites: number of adsorption cages to consider
    site_volume: volume of each adsorption site (A^3)
    temperature: units in Kelvin

Returns: working capacity / n_sites
"""
function compute_working_capacity(P_H::Float64, P_L::Float64, energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64)
    high_P_soln = exact_soln(P_H, energy_params, n_sites, site_volume, temperature)
    low_P_soln = exact_soln(P_L, energy_params, n_sites, site_volume, temperature)
    return high_P_soln["n"] - low_P_soln["n"]
end

"""
Return derivative of the working capacity between pressure P_H and P_L with respect to an energetic parameter.

Parameters:
    param: one of the energetic parameters
    P_H: high pressure in pressure-swing adsorption process
    P_L: low pressure in pressure-swing adsorption process
    baseline_energy_params: keep two params fixed and vary others.
"""
function optimize_working_capacity(param::AbstractString, P_H::Float64, P_L::Float64, 
        baseline_energy_params::EnergyParams, n_sites::Int, site_volume::Float64, temperature::Float64)

    @assert(param in ["ϵ_0", "ϵ", "ϵ_l"], "Not a viable parameter")

    # store optimal energy params here.
    opt_energy_params = deepcopy(baseline_energy_params)

    """
    We seek to make this function zero to optimize the working capacity WRT one
    of the energetic parameters. This function returns the derivative of the working
    capacity depending upon the value of the energetic parameter we are interested in.
    """
    function derivative_of_working_capacity(param_value)
        # modify the energetic parameter of interest
        if param == "ϵ_0"
            opt_energy_params.ϵ_0 = param_value
        elseif param == "ϵ"
            opt_energy_params.ϵ = param_value
        elseif param == "ϵ_l"
            opt_energy_params.ϵ_l = param_value
        end

        # Compute solution at P_H and P_L with these energetic params
        high_P_soln = exact_soln(P_H, opt_energy_params, n_sites, site_volume, temperature)
        low_P_soln = exact_soln(P_L, opt_energy_params, n_sites, site_volume, temperature)

        # get derivative of working capacity
        dwc_dparam = high_P_soln["dn_dβ" * param] - low_P_soln["dn_dβ" * param]
        return dwc_dparam
    end
    
    # find parameter such that derivative of working capacity wrt that param is zero
    opt_param = fzero(derivative_of_working_capacity, -20.0, 50.0)
    
    # compute working capacity with these optimal energy params
    working_capacity = compute_working_capacity(P_H, P_L, opt_energy_params, n_sites, site_volume, temperature)

    return opt_energy_params, working_capacity
end

"""
Optimum energy of adsorption and fractional deliverable capacity for a Langmuir model.
http://pubs.rsc.org/en/content/articlehtml/2014/cp/c3cp55039g

Parameters:
    P_H: high pressure in pressure-swing adsorption process
    P_L: low pressure in pressure-swing adsorption process
    site_volume: A^3
    temperature: K
  
Returns: Dictionary with Uopt, Kopt, Dopt  
"""
function LangmuirOptU(P_H::Float64, P_L::Float64, site_volume::Float64, temperature::Float64)
    result = Dict{AbstractString, Float64}()
    result["Uopt"] = 8.314 * temperature / 1000.0 * log(sqrt(P_H * P_L) / (k_b * temperature) *  site_volume)
    result["Dopt"] = P_H / (sqrt(P_H * P_L) + P_H) - P_L / (sqrt(P_H * P_L) + P_L)
    result["Kopt"] = 1 / sqrt(P_H * P_L)
    return result
end
