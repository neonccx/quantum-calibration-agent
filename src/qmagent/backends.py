"""Explicit simulation backend selection; no hardware adapter is registered."""

def make_backend(settings, seed=None):
    selected_seed = settings.seed if seed is None else seed
    if settings.backend == "physical":
        from .physical_backend import PhysicalSimulator
        return PhysicalSimulator(selected_seed, settings.noise_scale, settings.simulation_profile)
    if settings.backend == "legacy":
        from .simulator import AnalyticSimulator
        return AnalyticSimulator(selected_seed, settings.noise_scale)
    raise ValueError("Unknown simulation backend")
