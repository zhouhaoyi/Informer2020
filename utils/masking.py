"""Create types of masks to be used in various places in transformers.

- Full mask (any key masked for any query)
- Length mask (masking out everything after a length)
- Triangular causal mask (mask any key succeeding the query)

All mask implementations should provide a single interface to be used by the
transformer layers and the attention layers.

NOTE: In all cases the value 1 or True signifies what should be kept and not
      what should be deleted/masked.
"""

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
                # make sure that the mask starts with 1s and continues with 0s
                # this should be changed to something more efficient, however,
                # I chose simplicity over efficiency since the LengthMask class
                # will be used anyway (and the result is cached)
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
    """Thin wrapper over a pytorch tensor that provides the BaseMask
    interface.

    The arguments can be given both by keyword arguments and positional
    arguments. To imitate function overloading, the constructor checks the type
    of the first argument and if it is a tensor it treats it as the mask.
    otherwise it assumes that it was the N argument.

    Arguments
    ---------
        mask: The mask as a PyTorch tensor.
        N: The rows of the all True mask to be created if the mask argument is
           not provided.
        M: The columns of the all True mask to be created if the mask argument
           is not provided. If N is given M defaults to N.
        device: The device to create the mask in (defaults to cpu)
    """
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
    """Provide a BaseMask interface for lengths. Mostly to be used with
    sequences of different lengths.
    
    Arguments
    ---------
        lengths: The lengths as a PyTorch long tensor
        max_len: The maximum length for the mask (defaults to lengths.max())
        device: The device to be used for creating the masks (defaults to
                lengths.device)
    """
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
    """A square matrix with everything masked out above the diagonal.
    
    Arguments
    ---------
        N: The size of the matrix
        device: The device to create the mask in (defaults to cpu)
    """
    def __init__(self, N, device="cpu"):
        lengths = torch.arange(1, N+1, device=device)
        super(TriangularCausalMask, self).__init__(lengths, N, device)
        self._lower_triangular = True
