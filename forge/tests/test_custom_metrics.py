import pytest
import torch
import torch.nn.functional as F
import nequip.data.stats
from forge.workflows.allegro_utils.custom_metrics import (
    TailMSE,
    TailHuberLoss,
    FocalMSELoss,
    ForceAngleLoss,
    StressShearMAE,
    StressAngleLoss,
)

# Constants for testing
TOLERANCE = 1e-6

# --- Test Fixtures ---

@pytest.fixture(params=[torch.float32, torch.float64])
def dtype_and_monkeypatch(request, monkeypatch):
    """A pytest fixture that parameterizes tests for both float32 and float64.
    
    It also uses monkeypatch to set the nequip global dtype for the duration
    of the test, ensuring that metrics behave as they would in a real
    training environment of that precision.
    """
    dtype = request.param
    monkeypatch.setattr(nequip.data.stats, "_GLOBAL_DTYPE", dtype)
    return dtype

@pytest.fixture
def a_range_of_errors(dtype_and_monkeypatch):
    """Provides a tensor with a wide range of errors."""
    return torch.linspace(-10, 10, 100, dtype=dtype_and_monkeypatch)

@pytest.fixture
def force_tensors(dtype_and_monkeypatch):
    """Provides prediction and target tensors for force-like data."""
    dtype = dtype_and_monkeypatch
    pred = torch.randn(10, 3, dtype=dtype)
    target = torch.randn(10, 3, dtype=dtype)
    return pred, target

@pytest.fixture
def stress_tensors(dtype_and_monkeypatch):
    """Provides prediction and target tensors for stress-like data."""
    dtype = dtype_and_monkeypatch
    pred = torch.randn(5, 3, 3, dtype=dtype)
    target = torch.randn(5, 3, 3, dtype=dtype)
    return pred, target
    
# --- Test Cases ---

def test_tail_mse_per_batch(a_range_of_errors):
    """Tests that TailMSE correctly computes MSE on the top 10% of errors in a batch."""
    metric = TailMSE(quantile=0.9)
    pred = a_range_of_errors
    target = torch.zeros_like(pred)
    
    metric.update(pred, target)
    result = metric.compute()

    abs_err = torch.abs(pred)
    threshold = torch.quantile(abs_err.to(torch.float32), 0.9).to(abs_err.device)
    tail_mask = abs_err >= threshold
    expected_tail_errs = pred[tail_mask]
    expected_result = torch.mean(expected_tail_errs.pow(2))

    torch.testing.assert_close(result, expected_result, atol=TOLERANCE, rtol=TOLERANCE)

def test_tail_huber_loss_per_batch(force_tensors):
    """Tests that TailHuberLoss correctly computes Huber loss on the tail of force errors."""
    pred, target = force_tensors
    err_norm = torch.linalg.norm(pred - target, dim=-1)
    
    metric = TailHuberLoss(quantile=0.8, delta=0.5)
    metric.update(pred, target)
    result = metric.compute()
    
    threshold = torch.quantile(err_norm.to(torch.float32), 0.8).to(err_norm.device)
    tail_mask = err_norm >= threshold
    expected_preds = pred[tail_mask]
    expected_targets = target[tail_mask]
    
    expected_result = torch.mean(F.huber_loss(expected_preds, expected_targets, delta=0.5, reduction='none'))
    
    torch.testing.assert_close(result, expected_result, atol=TOLERANCE, rtol=TOLERANCE)

def test_focal_mse_loss(a_range_of_errors):
    """Tests that FocalMSELoss down-weights smaller errors."""
    metric_focal = FocalMSELoss(beta=1.0, gamma=2.0)
    metric_mse = TailMSE(quantile=0.0) # Standard MSE
    
    pred = a_range_of_errors
    target = torch.zeros_like(pred)
    
    metric_focal.update(pred, target)
    focal_result = metric_focal.compute()
    
    metric_mse.update(pred, target)
    mse_result = metric_mse.compute()
    
    assert focal_result < mse_result

def test_force_angle_loss(force_tensors):
    """Tests ForceAngleLoss for different alignment cases."""
    metric = ForceAngleLoss()
    pred, target = force_tensors
    dtype = pred.dtype # Get dtype from parameterized fixture

    # Run a simple case
    metric.update(pred, target)
    result = metric.compute()
    assert result.dtype == dtype

    # Run a specific case for correctness
    metric.reset()
    pred_case = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=dtype)
    target_case = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype)
    
    metric.update(pred_case, target_case)
    result = metric.compute()
    torch.testing.assert_close(result, torch.tensor(1.0, dtype=dtype), atol=TOLERANCE, rtol=0)

def test_stress_shear_mae(stress_tensors):
    """Tests that StressShearMAE only considers off-diagonal elements."""
    metric = StressShearMAE()
    pred, target = stress_tensors
    metric.update(pred, target)
    result = metric.compute()

    offdiag_indices = [(0, 1), (0, 2), (1, 2)]
    abs_errs = []
    for i, j in offdiag_indices:
        abs_errs.append(torch.abs(pred[:, i, j] - target[:, i, j]))
    
    mean_abs_err = torch.mean(torch.stack(abs_errs, dim=0), dim=0)
    expected_result = torch.mean(mean_abs_err)
    
    torch.testing.assert_close(result, expected_result, atol=TOLERANCE, rtol=TOLERANCE)

def test_stress_angle_loss(stress_tensors):
    """Tests the StressAngleLoss calculation."""
    metric = StressAngleLoss()
    pred, target = stress_tensors
    metric.update(pred, target)
    result = metric.compute()

    def to_voigt(stress):
        return torch.stack([
            stress[:, 0, 0], stress[:, 1, 1], stress[:, 2, 2],
            stress[:, 1, 2], stress[:, 0, 2], stress[:, 0, 1]
        ], dim=-1)

    voigt_p = to_voigt(pred)
    voigt_t = to_voigt(target)
    cos_phi = (voigt_p * voigt_t).sum(-1) / (voigt_p.norm(dim=-1) * voigt_t.norm(dim=-1) + 1e-8)
    expected_result = torch.mean(torch.tensor(1.0, dtype=pred.dtype, device=pred.device) - cos_phi)
    
    torch.testing.assert_close(result, expected_result, atol=TOLERANCE, rtol=TOLERANCE)

def test_running_mean_metrics(dtype_and_monkeypatch):
    """Tests the running mean behavior of _MeanX-based metrics over multiple batches."""
    dtype = dtype_and_monkeypatch
    metric = StressShearMAE()
    
    pred1, target1 = torch.randn(5, 3, 3, dtype=dtype), torch.randn(5, 3, 3, dtype=dtype)
    metric.update(pred1, target1)
    
    pred2, target2 = torch.randn(5, 3, 3, dtype=dtype), torch.randn(5, 3, 3, dtype=dtype)
    metric.update(pred2, target2)
    
    result = metric.compute()
    
    metric_total = StressShearMAE()
    metric_total.update(torch.cat([pred1, pred2]), torch.cat([target1, target2]))
    expected_result = metric_total.compute()

    torch.testing.assert_close(result, expected_result, atol=TOLERANCE, rtol=TOLERANCE) 