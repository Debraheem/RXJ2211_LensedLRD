"""Check notebook blackbody normalization without executing notebook workflows.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_blackbody_normalization.py' -v

Requires NumPy and Astropy. Only the two named function definitions are loaded;
catalogue access, plotting, fitting, and notebook top-level statements are skipped.
"""

import ast
import json
from pathlib import Path
import unittest

import numpy as np
from astropy import constants as const, units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.modeling.models import BlackBody


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "Fig2ab_Fig3abc_FigS10a_lc_fitting.ipynb": "blackbody_Fnu",
    "FigS6_S10_galfitm_sed_RX2lc.ipynb": "blackbody_Fnu",
    "check_lensmodel_systematics.ipynb": "blackbody_fnu_njy",
    "check_z6magnification.ipynb": "blackbody_Fnu",
    "lc_fitting_chemamodel.ipynb": "blackbody_Fnu",
    "lc_othersources.ipynb": "blackbody_Fnu",
}
TEMPERATURES = (3000.0, 4000.0, 6000.0)
REDSHIFTS = (0.1, 1.0, 4.3, 4.8, 6.0)


def load_blackbody(name, pure_blackbody=True, root=ROOT):
    """Load definitions only; replace Balmer attenuation by unity for flux tests."""
    path = root / "analysis_code" / name
    notebook = json.loads(path.read_text())
    function_name = NOTEBOOKS[name]
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and f"def {function_name}(" in "".join(cell.get("source", []))
    ]
    if len(sources) != 1:
        raise ValueError(f"Expected one {function_name} definition in {path}")
    nodes = [
        node for node in ast.parse(sources[0]).body
        if isinstance(node, ast.FunctionDef)
        and node.name in (function_name, "balmer_step")
    ]
    namespace = {
        "np": np, "const": const, "u": u, "cosmo": cosmo,
        "lambda_balmer": 3646e-4, "lambda_balmer_um": 3646e-4,
        "Z_SRC": 4.3,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    if pure_blackbody:
        namespace["balmer_step"] = lambda wave, z: np.ones_like(wave)
    return namespace[function_name]


class BlackbodyNormalizationTests(unittest.TestCase):
    def test_integrated_observer_flux(self):
        # Multiple redshifts matter: the two old errors cancel at z=1.
        # A temperature-scaled frequency grid resolves the whole Planck spectrum.
        x = np.geomspace(1e-6, 1e3, 40000)
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        for name in NOTEBOOKS:
            blackbody = load_blackbody(name)
            for temperature in TEMPERATURES:
                nu_rest = x * const.k_B.cgs.value * temperature / const.h.cgs.value
                for redshift in REDSHIFTS:
                    with self.subTest(notebook=name, T=temperature, z=redshift):
                        nu_obs = nu_rest / (1 + redshift)
                        wave_um = const.c.cgs.value / nu_obs * 1e4
                        with np.errstate(over="ignore", under="ignore"):
                            fnu = blackbody(wave_um, temperature, 44.0, redshift) * 1e-32
                        self.assertTrue(np.all(np.isfinite(fnu)))
                        distance = cosmo.luminosity_distance(redshift).to_value(u.cm)
                        recovered = 4 * np.pi * distance**2 * integrate(fnu, nu_obs)
                        np.testing.assert_allclose(recovered / 1e44, 1.0, rtol=1e-6)

    def test_independent_observer_temperature_spectrum(self):
        # An observer-frame blackbody shape normalized to Fbol uses neither
        # notebook surface area nor its spectral redshift conversion.
        intensity_unit = u.erg / u.s / u.cm**2 / u.Hz / u.sr
        for name in NOTEBOOKS:
            blackbody = load_blackbody(name)
            for temperature in TEMPERATURES:
                for redshift in REDSHIFTS:
                    with self.subTest(notebook=name, T=temperature, z=redshift):
                        observed_temperature = temperature / (1 + redshift)
                        nu_obs = (
                            np.geomspace(0.01, 30, 150)
                            * const.k_B.cgs.value * observed_temperature / const.h.cgs.value
                        )
                        intensity = BlackBody(observed_temperature * u.K)(
                            nu_obs * u.Hz
                        ).to_value(intensity_unit)
                        distance = cosmo.luminosity_distance(redshift).to_value(u.cm)
                        fbol = 1e44 / (4 * np.pi * distance**2)
                        expected = fbol * np.pi * intensity / (
                            const.sigma_sb.cgs.value * observed_temperature**4
                        )
                        wave_um = const.c.cgs.value / nu_obs * 1e4
                        actual = blackbody(wave_um, temperature, 44.0, redshift) * 1e-32
                        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=0)

    def test_luminosity_scaling_with_original_balmer_prescription(self):
        wave_um = np.geomspace(0.5, 20.0, 200)
        for name in NOTEBOOKS:
            blackbody = load_blackbody(name, pure_blackbody=False)
            for redshift in REDSHIFTS:
                with self.subTest(notebook=name, z=redshift):
                    # The existing sigmoid can overflow to its finite limiting
                    # value far from the Balmer edge; keep that prescription.
                    with np.errstate(over="ignore"):
                        faint = blackbody(wave_um, 4000.0, 43.0, redshift)
                        bright = blackbody(wave_um, 4000.0, 45.0, redshift)
                    self.assertTrue(np.all(np.isfinite(bright)))
                    np.testing.assert_allclose(bright, 100 * faint, rtol=1e-12, atol=0)


if __name__ == "__main__":
    unittest.main()
