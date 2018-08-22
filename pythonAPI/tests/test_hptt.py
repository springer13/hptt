import pytest
from numpy.testing import assert_allclose

import numpy as np
import hptt


class TestTranpose:

    @pytest.mark.parametrize("dtype", [
        'float32',
        'float64',
        'complex64',
        'complex128',
    ])
    @pytest.mark.parametrize("axes", [
        (0, 1, 2, 3),
        (3, 1, 2, 0),
        (1, 0, 2, 3),
        None,
    ])
    @pytest.mark.parametrize("order", [
        'C',
        'F',
    ])
    def test_against_numpy(self, dtype, axes, order):

        X = np.random.randn(3, 4, 5, 6)
        if 'complex' in dtype:
            X = X + 1.0j * np.random.randn(3, 4, 5, 6)

        X = X.astype(dtype)

        if order == 'F':
            X = np.asfortranarray(X)

        XT_numpy = np.transpose(X, axes)
        XT_hptt = hptt.transpose(X, axes)

        assert_allclose(XT_numpy, XT_hptt)

        assert (XT_hptt.flags['C_CONTIGUOUS'] if order == 'C' else
                XT_hptt.flags['F_CONTIGUOUS'])
