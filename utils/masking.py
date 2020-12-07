import torch

class BaseMask(object):
    @property
    def bool_matrix(self):
        """Return a bool (uint8) matrix with 1s to all places that should be
        kept."""
        raise NotImplementedError()

    @property
    def float_matrix(self):
        """Return the bool matrix as a float to be used as a multiplicative
        mask for non softmax attentions."""
        if not hasattr(self, "_float_matrix"):
            with torch.no_grad():
                self._float_matrix = self.bool_matrix.float()
        return self._float_matrix

    @property
    def lengths(self):
        """If the matrix is of the following form
        
            1 1 1 0 0 0 0
            1 0 0 0 0 0 0
            1 1 0 0 0 0 0

        then return it as a vector of integers

            3 1 2.
        """
        if not hasattr(self, "_lengths"):
            with torch.no_grad():
                lengths = self.bool_matrix.long().sum(dim=-1)
                m = self.bool_matrix.view(-1, self.shape[-1])
                for i, l in enumerate(lengths.view(-1)):
                    if not torch.all(m[i, :l]):
                        raise ValueError("The mask is not a length mask")
                self._lengths = lengths
        return self._lengths

    @property
    def shape(self):
        """Return the shape of the boolean mask."""
        return self.bool_matrix.shape

    @property
    def additive_matrix(self):
        """Return a float matrix to be added to an attention matrix before
        softmax."""
        if not hasattr(self, "_additive_matrix"):
            with torch.no_grad():
                self._additive_matrix = torch.log(self.bool_matrix.float())
        return self._additive_matrix

    @property
    def additive_matrix_finite(self):
        """Same as additive_matrix but with -1e24 instead of infinity."""
        if not hasattr(self, "_additive_matrix_finite"):
            with torch.no_grad():
                self._additive_matrix_finite = (
                    (~self.bool_matrix).float() * (-1e24)
                )
        return self._additive_matrix_finite

    @property
    def all_ones(self):
        """Return true if the mask is all ones."""
        if not hasattr(self, "_all_ones"):
            with torch.no_grad():
                self._all_ones = torch.all(self.bool_matrix)
        return self._all_ones

    @property
    def lower_triangular(self):
        """Return true if the attention is a triangular causal mask."""
        if not hasattr(self, "_lower_triangular"):
            self._lower_triangular = False
            with torch.no_grad():
                try:
                    lengths = self.lengths
                    if len(lengths.shape) == 1:
                        target = torch.arange(
                            1,
                            len(lengths)+1,
                            device=lengths.device
                        )
                        self._lower_triangular = torch.all(lengths == target)
                except ValueError:
                    pass
        return self._lower_triangular


class FullMask(BaseMask):
    def __init__(self, mask=None, N=None, M=None, device="cpu"):
        # mask is a tensor so we ignore N and M
        if mask is not None and isinstance(mask, torch.Tensor):
            if mask.dtype != torch.bool:
                raise ValueError("FullMask expects the mask to be bool")
            with torch.no_grad():
                self._mask = mask.clone()
            return

        # mask is an integer, N is an integer and M is None so assume they were
        # passed as N, M
        if mask is not None and M is None and isinstance(mask, int):
            M = N
            N = mask

        if N is not None:
            M = M or N
            with torch.no_grad():
                self._mask = torch.ones(N, M, dtype=torch.bool, device=device)
            self._all_ones = True
            return

        raise ValueError("Either mask or N should be provided")

    @property
    def bool_matrix(self):
        return self._mask


class LengthMask(BaseMask):
    def __init__(self, lengths, max_len=None, device=None):
        self._device = device or lengths.device
        with torch.no_grad():
            self._lengths = lengths.clone().to(self._device)
        self._max_len = max_len or self._lengths.max()

        self._bool_matrix = None

    @property
    def bool_matrix(self):
        if self._bool_matrix is None:
            with torch.no_grad():
                indices = torch.arange(self._max_len, device=self._device)
                self._bool_matrix = (
                    indices.view(1, -1) < self._lengths.view(-1, 1)
                )
        return self._bool_matrix


class TriangularCausalMask(LengthMask):
    def __init__(self, N, device="cpu"):
        lengths = torch.arange(1, N+1, device=device)
        super(TriangularCausalMask, self).__init__(lengths, N, device)
        self._lower_triangular = True

class ProbMask(LengthMask):
    def __init__(self, B, H, L, index, scores, device="cpu"):
        lengths = torch.arange(1, L+1, device=device)
        super(ProbMask, self).__init__(lengths, L, device)
        
        _mask = torch.ones(L, scores.shape[-1]).to(device).triu(1).byte()
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._bool_matrix = (indicator.view(scores.shape)<=0).to(device)
    
    @property
    def bool_matrix(self):
        return self._bool_matrix