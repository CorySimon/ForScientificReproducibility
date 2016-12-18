# Define Boltzmann constant so we can work with units:
#    Pressure: bar
#    Volume: A^3
#    Temperature: K
const k_b = 1.380648e-23 / 100000.0 * 1e30
# J/K = Pa-m3 / K -> bar-A3 / K

# Store energetic parameters here.
#  All units are in kJ/mol.
type EnergyParams
    ϵ_l::Float64  # penalty for linker sticking into pore
    ϵ::Float64  # benefit for adsorbate given by linker sticking into pore
    ϵ_0::Float64 # background energy of adsorption for adsorbate, independent of surrounding linker configurations
end

function construct_energy_param(;ϵ_l::Float64=NaN, ϵ::Float64=NaN, ϵ_0::Float64=NaN)
    return EnergyParams(ϵ_l, ϵ, ϵ_0)
end

function print_energy_params(energy_params::EnergyParams)
    @printf("eps_0 = %f kJ/mol (+ve is favorable)\n", energy_params.ϵ_0)
    @printf("eps_l = %f kJ/mol (+ve is unfavorable)\n", energy_params.ϵ_l)
    @printf("eps = %f kJ/mol (+ve is favorable)\n", energy_params.ϵ)
end
