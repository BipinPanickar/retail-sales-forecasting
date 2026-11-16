"""SARIMA(p,d,q)(P,D,Q,s). The seasonal period s comes from config AFTER inspecting the data
(ACF/PACF/decomposition in the EDA notebook); it is not assumed silently."""
import itertools

from src.models.arima_model import SarimaxForecaster, choose_d, select_by_aic_then_validation


class SarimaForecaster(SarimaxForecaster):
    name = "sarima"

    @staticmethod
    def select(train, val, horizon, season_length, cfg_sarima, quick=False):
        d = choose_d(train["sales"])
        if quick:
            ps, qs, Ps, Qs, Ds = [0, 1], [0, 1], [0, 1], [0, 1], [1]
        else:
            ps = range(cfg_sarima["max_p"] + 1)
            qs = range(cfg_sarima["max_q"] + 1)
            Ps, Qs, Ds = cfg_sarima["P_candidates"], cfg_sarima["Q_candidates"], cfg_sarima["D_candidates"]
        cands = [((p, d, q), (P, D, Q, season_length))
                 for p, q, P, Q, D in itertools.product(ps, qs, Ps, Qs, Ds)]
        return select_by_aic_then_validation(cands, train, val, horizon, name="sarima")
