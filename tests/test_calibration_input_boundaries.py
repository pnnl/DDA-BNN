"""Reject partial draws and invalid numbers before replacing calibration files."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from training import temp_tune


def gaussian():
    return pd.DataFrame({
        'mu_lat_mean_A': [0.1, 0.2], 'mu_lat_mean_B': [0.3, 0.4],
        'std_ale_lat_A': [0.1, 0.2], 'std_epi_lat_B': [0.2, 0.3],
    })


def posterior():
    return pd.DataFrame({
        f'{prefix}{draw}_{target}': [0.1, 0.2]
        for draw in (0, 1) for target in ('A', 'B')
        for prefix in ('mu_lat_s', 'std_ale_lat_s')
    })


@pytest.mark.parametrize('change', [
    lambda df: df.drop(columns='mu_lat_s1_B'),
    lambda df: df.drop(columns='std_ale_lat_s0_B'),
    lambda df: df.rename(columns={c: c.replace('s1_', 's2_') for c in df}),
    lambda df: df.rename(columns={c: c.replace('s0_', 's3_') for c in df}),
    lambda df: df.assign(mu_lat_s999999999999999999999999999999999999999_A=0.1),
    lambda df: df.assign(mu_lat_s01_A=0.1),
    lambda df: df.assign(**{'mu_lat_s-1_A': 0.1}),
    lambda df: df.assign(mu_lat_swrong_A=0.1),
    lambda df: gaussian().assign(mu_lat_s0_A=0.1),
])
def test_incomplete_or_noncontiguous_posterior_draws_are_rejected(change):
    with pytest.raises(ValueError, match='posterior'):
        temp_tune.load_latent_components(change(posterior()), ['A', 'B'])


def test_complete_posterior_draws_and_underscored_targets_are_preserved():
    df = posterior().rename(columns=lambda c: c[:-1] + 'A_X' if c.endswith('_A') else c)
    mode, means, stds = temp_tune.load_latent_components(df[df.columns[::-1]], ['A_X', 'B'])
    assert mode == 'posterior' and means.shape == stds.shape == (2, 2, 2)
    np.testing.assert_array_equal(means[:, :, 0], [[0.1, 0.2], [0.1, 0.2]])


@pytest.mark.parametrize('make', [gaussian, posterior])
def test_empty_apply_data_is_rejected(make):
    with pytest.raises(ValueError, match='empty'):
        temp_tune.load_latent_components(make().iloc[:0], ['A', 'B'])


@pytest.mark.parametrize('column', ['mu_lat_mean_A', 'std_ale_lat_A', 'std_epi_lat_B'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_gaussian_components_must_be_finite(column, value):
    df = gaussian(); df.loc[0, column] = value
    with pytest.raises(ValueError, match='finite'):
        temp_tune.load_latent_components(df, ['A', 'B'])


@pytest.mark.parametrize('column', ['std_ale_lat_A', 'std_epi_lat_B'])
def test_gaussian_standard_deviations_cannot_be_negative(column):
    df = gaussian(); df.loc[0, column] = -0.1
    with pytest.raises(ValueError, match='nonnegative'):
        temp_tune.load_latent_components(df, ['A', 'B'])


@pytest.mark.parametrize('column', ['mu_lat_s1_A', 'std_ale_lat_s1_B'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_posterior_components_must_be_finite(column, value):
    df = posterior(); df.loc[0, column] = value
    with pytest.raises(ValueError, match='finite'):
        temp_tune.load_latent_components(df, ['A', 'B'])


def test_posterior_standard_deviations_cannot_be_negative():
    df = posterior(); df.loc[0, 'std_ale_lat_s1_B'] = -0.1
    with pytest.raises(ValueError, match='nonnegative'):
        temp_tune.load_latent_components(df, ['A', 'B'])


@pytest.mark.parametrize('mu, std, truth', [
    (np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))),
    (np.zeros((2, 2)), -np.ones((2, 2)), np.zeros((2, 2))),
    (np.full((2, 2), np.nan), np.ones((2, 2)), np.zeros((2, 2))),
    (np.zeros((2, 2)), np.full((2, 2), np.inf), np.zeros((2, 2))),
    (np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), np.nan)),
    (np.zeros((2, 2)), np.ones((1, 2)), np.zeros((2, 2))),
])
def test_tau_fit_rejects_invalid_arrays(mu, std, truth):
    with np.errstate(all='ignore'), pytest.raises(ValueError):
        temp_tune.fit_tau_per_target(mu, std, truth)


@pytest.mark.parametrize('failure', [
    'apply_nan', 'apply_negative', 'partial_posterior', 'apply_empty',
    'val_nan', 'val_negative', 'val_empty', 'K_zero', 'L_zero', 'K_bool', 'L_fraction', 'seed_negative', 'seed_fraction', 'fit_failure', 'sampling_failure',
])
def test_invalid_input_preserves_existing_tau_and_output(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(temp_tune.cfg, 'TARGETS', ['A', 'B'])
    (tmp_path / 'config_used.yaml').write_text('{}\n')
    torch.save({'tf_info': {'A': 'none', 'B': 'none'}}, tmp_path / 'data_meta.pt')
    val = pd.DataFrame({f'{prefix}{t}': [0.1, 0.2]
                        for t in ('A', 'B')
                        for prefix in ('mu_lat_mean_', 'std_tot_lat_', 'true_')})
    app = gaussian(); options = {'K': 2, 'L': 2}
    if failure == 'apply_nan': app.loc[0, 'mu_lat_mean_A'] = np.nan
    elif failure == 'apply_negative': app.loc[0, 'std_ale_lat_A'] = -0.1
    elif failure == 'partial_posterior': app['mu_lat_s0_A'] = 0.1
    elif failure == 'apply_empty': app = app.iloc[:0]
    elif failure == 'val_nan': val.loc[0, 'true_A'] = np.nan
    elif failure == 'val_negative': val.loc[0, 'std_tot_lat_A'] = -0.1
    elif failure == 'val_empty': val = val.iloc[:0]
    elif failure == 'K_zero': options['K'] = 0
    elif failure == 'L_zero': options['L'] = 0
    elif failure == 'K_bool': options['K'] = True
    elif failure == 'L_fraction': options['L'] = 1.5
    elif failure == 'seed_negative': options['seed'] = -1
    elif failure == 'seed_fraction': options['seed'] = 1.5
    elif failure == 'fit_failure':
        monkeypatch.setattr(temp_tune, 'minimize_scalar', lambda *a, **k:
                            SimpleNamespace(success=False, fun=0.0, x=0.0))
    elif failure == 'sampling_failure':
        def fail_after_fitting(*args, **kwargs):
            raise ValueError('simulated sampling failure')
        monkeypatch.setattr(temp_tune, 'sample_phys_gaussian', fail_after_fitting)
    val.to_csv(tmp_path / 'predictions_val.csv', index=False)
    app.to_csv(tmp_path / 'predictions_test.csv', index=False)
    tau_text = json.dumps({'A': 2.0, 'B': 3.0}) + '\n'
    output_text = 'preserve existing calibrated results\n'
    (tmp_path / 'taus.json').write_text(tau_text)
    output = tmp_path / 'predictions_test_cal.csv'; output.write_text(output_text)
    with pytest.raises(ValueError):
        temp_tune.run_temp_tune(tmp_path, **options)
    assert (tmp_path / 'taus.json').read_text() == tau_text
    assert output.read_text() == output_text


@pytest.mark.parametrize('result', [
    SimpleNamespace(success=False, fun=0.0, x=0.0),
    SimpleNamespace(success=True, fun=np.inf, x=0.0),
    SimpleNamespace(success=True, fun=0.0, x=np.nan),
])
def test_failed_or_nonfinite_optimizer_result_is_rejected(monkeypatch, result):
    monkeypatch.setattr(temp_tune, 'minimize_scalar', lambda *a, **k: result)
    with pytest.raises(ValueError, match='calibration failed'):
        temp_tune.fit_tau_per_target(np.zeros((2, 1)), np.ones((2, 1)), np.zeros((2, 1)))


def test_unconfigured_target_columns_do_not_change_selected_draws():
    df = posterior().assign(mu_lat_s7_unconfigured=0.1)
    mode, means, stds = temp_tune.load_latent_components(df, ['A', 'B'])
    assert mode == 'posterior' and means.shape == stds.shape == (2, 2, 2)
