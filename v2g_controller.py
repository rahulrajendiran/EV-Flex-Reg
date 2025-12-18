def soc_optimization_factor(
    soc,
    soc_min=0.2,
    soc_full=0.5,
    soc_max=0.9
):
    """
    SoC-aware V2G optimization curve.
    Returns a scaling factor in [0, 1].
    """
    if soc <= soc_min:
        return 0.0

    # Ramp-up region (protect low SoC)
    if soc_min < soc < soc_full:
        return (soc - soc_min) / (soc_full - soc_min)

    # Safe operating region
    if soc_full <= soc <= soc_max:
        return 1.0

    # High-SoC tapering (battery stress zone)
    return 0.8

def battery_degradation_cost_per_kwh(
    battery_cost_per_kwh=12000,   # ₹/kWh
    battery_capacity_kwh=60,      # kWh
    cycle_life=3000
):
    """
    Returns degradation cost in ₹ per kWh of discharged energy.
    """
    lifetime_throughput = battery_capacity_kwh * cycle_life
    total_battery_cost = battery_capacity_kwh * battery_cost_per_kwh
    return total_battery_cost / lifetime_throughput

def v2g_controller(
    flexible_kw,
    grid_price,
    soc,
    duration_hr=1.0,
    soc_min=0.2,
    export_limit=3.6
):
    """
    V2G controller with SoC-aware optimization and degradation cost.
    """

    # SoC-aware scaling
    soc_factor = soc_optimization_factor(soc, soc_min=soc_min)
    if soc_factor <= 0.0:
        return 0.0

    optimized_kw = min(flexible_kw * soc_factor, export_limit)

    # Energy-based evaluation
    energy_kwh = optimized_kw * duration_hr

    degradation_cost = (
        energy_kwh * battery_degradation_cost_per_kwh()
    )

    revenue = energy_kwh * grid_price

    # Economic decision
    if revenue > degradation_cost:
        return optimized_kw

    return 0.0
