"""
Comprehensive audit of IReD parameter units and conversions.
"""

import numpy as np

from israel_stewart.equations.ired_simple import HardSphereIReD

print("=" * 80)
print("COMPREHENSIVE IReD PARAMETER UNIT AUDIT")
print("=" * 80)

model = HardSphereIReD(temperature=0.4, cross_section=1.0)

# Define expected units for each parameter
params = {
    # Basic thermodynamic properties
    "temperature": ("GeV", model.temperature),
    "beta": ("GeV⁻¹", model.beta),
    "cross_section": ("fm²", model.cross_section),
    "particle_density": ("GeV³", model.particle_density),
    "energy_density": ("GeV⁴", model.energy_density),
    "pressure": ("GeV⁴", model.pressure),
    "entropy_density": ("GeV³", model.entropy_density),
    "mean_free_path": ("fm", model.mean_free_path),
    # First-order transport coefficients
    "shear_viscosity": ("GeV³", model.shear_viscosity()),
    "bulk_viscosity": ("GeV³", model.bulk_viscosity()),
    "diffusion_coefficient": ("GeV²", model.diffusion_coefficient()),
}

# Relaxation times (check ALL unit options)
relax_times = {
    "shear_relaxation_time (fm/c)": ("fm/c", model.shear_relaxation_time(time_unit="fm/c")),
    "shear_relaxation_time (natural)": ("GeV⁻¹", model.shear_relaxation_time(time_unit="natural")),
    "bulk_relaxation_time (fm/c)": ("fm/c", model.bulk_relaxation_time(time_unit="fm/c")),
    "bulk_relaxation_time (natural)": ("GeV⁻¹", model.bulk_relaxation_time(time_unit="natural")),
    "diffusion_relaxation_time (fm/c)": ("fm/c", model.diffusion_relaxation_time(time_unit="fm/c")),
    "diffusion_relaxation_time (natural)": (
        "GeV⁻¹",
        model.diffusion_relaxation_time(time_unit="natural"),
    ),
}

# Second-order coefficients (pure time)
second_order_pure_time = {
    "tau_pi_pi (fm/c)": ("fm/c", model.tau_pi_pi(time_unit="fm/c")),
    "tau_pi_pi (natural)": ("GeV⁻¹", model.tau_pi_pi(time_unit="natural")),
    "lambda_V_V (fm/c)": ("fm/c", model.lambda_V_V(time_unit="fm/c")),
    "lambda_V_V (natural)": ("GeV⁻¹", model.lambda_V_V(time_unit="natural")),
}

# Second-order coefficients (pure energy, NO time)
# NOTE: λ_πV has units GeV¹ (energy, not time!)
# From IReD Table IV: λ_πn = 0.20890/β = 0.20890 × T
second_order_energy = {
    "lambda_pi_V": ("GeV¹", model.lambda_pi_V()),
}

# Second-order coefficients (mixed units with time)
second_order_mixed = {
    "lambda_V_pi (fm/c)": ("GeV⁻²·fm/c", model.lambda_V_pi(time_unit="fm/c")),
    "lambda_V_pi (natural)": ("GeV⁻³", model.lambda_V_pi(time_unit="natural")),
    "ell_V_pi (fm/c)": ("GeV⁻²·fm/c", model.ell_V_pi(time_unit="fm/c")),
    "ell_V_pi (natural)": ("GeV⁻³", model.ell_V_pi(time_unit="natural")),
    "tau_V_pi (fm/c)": ("GeV⁻⁵·fm/c", model.tau_V_pi(time_unit="fm/c")),
    "tau_V_pi (natural)": ("GeV⁻⁶", model.tau_V_pi(time_unit="natural")),
}

# Second-order coefficients (no time dimension)
second_order_no_time = {
    "delta_pi_pi": ("dimensionless", model.delta_pi_pi()),
    "delta_V_V": ("dimensionless", model.delta_V_V()),
    "ell_pi_V": ("GeV¹", model.ell_pi_V()),
    "tau_pi_V": ("GeV⁻³", model.tau_pi_V()),
}


def print_section(title, params_dict):
    print(f"\n{title}")
    print("-" * 80)
    for name, (units, value) in params_dict.items():
        print(f"{name:40s} = {value:12.6e}  [{units}]")


print_section("BASIC THERMODYNAMIC PROPERTIES", params)
print_section("RELAXATION TIMES (BOTH UNITS)", relax_times)
print_section("SECOND-ORDER: PURE TIME DIMENSION", second_order_pure_time)
print_section("SECOND-ORDER: PURE ENERGY (NO TIME!)", second_order_energy)
print_section("SECOND-ORDER: MIXED UNITS (TIME + ENERGY)", second_order_mixed)
print_section("SECOND-ORDER: NO TIME DIMENSION", second_order_no_time)

# Verify conversion consistency
print("\n" + "=" * 80)
print("UNIT CONVERSION VERIFICATION")
print("=" * 80)

HBARC = 0.1973269804  # GeV·fm

tests = [
    (
        "τ_π conversion",
        model.shear_relaxation_time(time_unit="fm/c") / HBARC,
        model.shear_relaxation_time(time_unit="natural"),
    ),
    (
        "τ_Π conversion",
        model.bulk_relaxation_time(time_unit="fm/c") / HBARC,
        model.bulk_relaxation_time(time_unit="natural"),
    ),
    (
        "τ_V conversion",
        model.diffusion_relaxation_time(time_unit="fm/c") / HBARC,
        model.diffusion_relaxation_time(time_unit="natural"),
    ),
    (
        "τ_ππ conversion",
        model.tau_pi_pi(time_unit="fm/c") / HBARC,
        model.tau_pi_pi(time_unit="natural"),
    ),
    # NOTE: λ_πV removed - it has units GeV¹ (energy), NOT time!
    # It doesn't have a fm/c vs natural conversion.
    (
        "λ_Vπ conversion",
        model.lambda_V_pi(time_unit="fm/c") / HBARC,
        model.lambda_V_pi(time_unit="natural"),
    ),
]

all_pass = True
for name, expected, actual in tests:
    error = abs(expected - actual) / abs(expected) if expected != 0 else abs(actual)
    status = "✓" if error < 1e-10 else "✗"
    all_pass = all_pass and (error < 1e-10)
    print(f"{status} {name:25s}: error = {error:.2e}")

print(f"\n{'✓ ALL CONVERSIONS CORRECT' if all_pass else '✗ SOME CONVERSIONS FAILED'}")
