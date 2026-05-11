"""
Unified interface for ROCKET family time series transforms.

This module provides sklearn-like transformer classes for:
- ROCKET: Random Convolutional Kernel Transform
- MiniROCKET: A faster, almost deterministic variant
- RASTER: ROCKET with segmented biases

References:
    - ROCKET: https://arxiv.org/abs/1910.13051
    - MiniROCKET: https://arxiv.org/abs/2012.08791
    - RASTER: https://doi.org/10.1109/MLSP55844.2023.10285973
"""

from typing import Optional, Tuple, Union
import numpy as np

# Import underlying implementations
from . import rocket as _rocket
from . import miniROCKET as _minirocket
from . import minirocket_multivariate as _minirocket_mv
from . import raster_multivariate as _raster_mv


class ROCKET:
    """
    ROCKET (RandOm Convolutional KErnel Transform) for time series.
    
    Transforms time series into features using random convolutional kernels.
    For each kernel, computes PPV (proportion of positive values) and max pooling.
    
    Parameters:
        num_kernels: Number of random kernels to generate. Default 10,000.
        random_state: Random seed for reproducibility. Default None.
    
    Attributes:
        kernels_: Fitted kernel parameters (weights, lengths, biases, dilations, paddings).
        n_features_out_: Number of output features (2 * num_kernels).
    
    Example:
        >>> rocket = ROCKET(num_kernels=10000)
        >>> rocket.fit(X_train)
        >>> X_train_features = rocket.transform(X_train)
        >>> X_test_features = rocket.transform(X_test)
    
    Reference:
        Dempster et al., "ROCKET: Exceptionally fast and accurate time series 
        classification using random convolutional kernels", 2020.
    """
    
    def __init__(
        self,
        num_kernels: int = 10_000,
        random_state: Optional[int] = None
    ):
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.kernels_ = None
        self.n_features_out_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "ROCKET":
        """
        Fit the ROCKET transform by generating random kernels.
        
        Args:
            X: Time series data, shape (n_samples, series_length).
            y: Ignored, present for sklearn compatibility.
        
        Returns:
            self: The fitted transformer.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = self._validate_input(X)
        _, input_length = X.shape
        
        self.kernels_ = _rocket.generate_kernels(input_length, self.num_kernels)
        self.n_features_out_ = self.num_kernels * 2  # PPV and max for each kernel
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform time series to ROCKET features.
        
        Args:
            X: Time series data, shape (n_samples, series_length).
        
        Returns:
            Features array, shape (n_samples, num_kernels * 2).
        """
        if self.kernels_ is None:
            raise RuntimeError("ROCKET must be fitted before transform")
        
        X = self._validate_input(X)
        return _rocket.apply_kernels(X, self.kernels_)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Time series data, shape (n_samples, series_length).
            y: Ignored, present for sklearn compatibility.
        
        Returns:
            Features array, shape (n_samples, num_kernels * 2).
        """
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and convert input to correct dtype."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        return X.astype(np.float64)
    
    def __repr__(self) -> str:
        return f"ROCKET(num_kernels={self.num_kernels})"


class MiniROCKET:
    """
    MiniROCKET: A very fast, almost deterministic transform for time series.
    
    A faster variant of ROCKET that uses a fixed set of kernel configurations
    with data-driven bias selection.
    
    Parameters:
        num_features: Number of features to generate. Default 10,000.
        max_dilations_per_kernel: Maximum dilations per kernel. Default 32.
        random_state: Random seed for reproducibility. Default None.
    
    Attributes:
        parameters_: Fitted parameters (dilations, num_features_per_dilation, biases).
        n_features_out_: Number of output features.
    
    Example:
        >>> minirocket = MiniROCKET(num_features=10000)
        >>> minirocket.fit(X_train)
        >>> X_train_features = minirocket.transform(X_train)
        >>> X_test_features = minirocket.transform(X_test)
    
    Reference:
        Dempster et al., "MiniRocket: A Very Fast (Almost) Deterministic 
        Transform for Time Series Classification", 2021.
    """
    
    def __init__(
        self,
        num_features: int = 10_000,
        max_dilations_per_kernel: int = 32,
        random_state: Optional[int] = None
    ):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state
        self.parameters_ = None
        self.n_features_out_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "MiniROCKET":
        """
        Fit the MiniROCKET transform.
        
        Args:
            X: Time series data, shape (n_samples, series_length).
            y: Ignored, present for sklearn compatibility.
        
        Returns:
            self: The fitted transformer.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = self._validate_input(X)
        
        # Fit returns: dilations, num_features_per_dilation, random_size, biases
        full_params = _minirocket.fit(
            X, 
            num_features=self.num_features,
            max_dilations_per_kernel=self.max_dilations_per_kernel
        )
        
        # Store as (dilations, num_features_per_dilation, biases) for transform
        dilations, num_features_per_dilation, _, biases = full_params
        self.parameters_ = (dilations, num_features_per_dilation, biases)
        self.n_features_out_ = 84 * np.sum(num_features_per_dilation)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform time series to MiniROCKET features.
        
        Args:
            X: Time series data, shape (n_samples, series_length).
        
        Returns:
            Features array, shape (n_samples, n_features_out_).
        """
        if self.parameters_ is None:
            raise RuntimeError("MiniROCKET must be fitted before transform")
        
        X = self._validate_input(X)
        return _minirocket.transform(X, self.parameters_, 'ter')
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Time series data, shape (n_samples, series_length).
            y: Ignored, present for sklearn compatibility.
        
        Returns:
            Features array, shape (n_samples, n_features_out_).
        """
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and convert input to correct dtype."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        return X.astype(np.float32)
    
    def __repr__(self) -> str:
        return f"MiniROCKET(num_features={self.num_features})"


class MiniROCKETMultivariate:
    """
    MiniROCKET for multivariate time series.
    
    Extends MiniROCKET to handle multiple channels/variables.
    
    Parameters:
        num_features: Number of features to generate. Default 10,000.
        max_dilations_per_kernel: Maximum dilations per kernel. Default 32.
        random_state: Random seed for reproducibility. Default None.
    
    Attributes:
        parameters_: Fitted parameters.
        n_features_out_: Number of output features.
    
    Example:
        >>> # X has shape (n_samples, n_channels, series_length)
        >>> minirocket_mv = MiniROCKETMultivariate(num_features=10000)
        >>> minirocket_mv.fit(X_train)
        >>> X_train_features = minirocket_mv.transform(X_train)
    """
    
    def __init__(
        self,
        num_features: int = 10_000,
        max_dilations_per_kernel: int = 32,
        random_state: Optional[int] = None
    ):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state
        self.parameters_ = None
        self.n_features_out_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "MiniROCKETMultivariate":
        """
        Fit the multivariate MiniROCKET transform.
        
        Args:
            X: Time series data, shape (n_samples, n_channels, series_length).
            y: Ignored, present for sklearn compatibility.
        
        Returns:
            self: The fitted transformer.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = self._validate_input(X)
        
        self.parameters_ = _minirocket_mv.fit(
            X,
            num_features=self.num_features,
            max_dilations_per_kernel=self.max_dilations_per_kernel
        )
        
        # Compute output features
        _, _, dilations, num_features_per_dilation, _ = self.parameters_
        self.n_features_out_ = 84 * np.sum(num_features_per_dilation)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform multivariate time series to features.
        
        Args:
            X: Time series data, shape (n_samples, n_channels, series_length).
        
        Returns:
            Features array, shape (n_samples, n_features_out_).
        """
        if self.parameters_ is None:
            raise RuntimeError("MiniROCKETMultivariate must be fitted before transform")
        
        X = self._validate_input(X)
        return _minirocket_mv.transform(X, self.parameters_)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and convert input to correct dtype."""
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (n_samples, n_channels, series_length), got {X.ndim}D")
        return X.astype(np.float32)
    
    def __repr__(self) -> str:
        return f"MiniROCKETMultivariate(num_features={self.num_features})"


class RASTER:
    """
    RASTER: ROCKET with segmented/refined biases.
    
    An extension of MiniROCKET that uses segmented thresholds (biases)
    for potentially better discriminative features.
    
    Parameters:
        num_features: Number of features to generate. Default 10,000.
        max_dilations_per_kernel: Maximum dilations per kernel. Default 32.
        num_segments: Maximum number of segments for biases. Default 4.
        fixed_segments: If True, use fixed segment size. Default False.
        random_state: Random seed for reproducibility. Default None.
    
    Attributes:
        parameters_: Fitted parameters.
        n_features_out_: Number of output features.
    
    Example:
        >>> raster = RASTER(num_features=10000, num_segments=4)
        >>> raster.fit(X_train)
        >>> X_train_features = raster.transform(X_train)
        >>> X_test_features = raster.transform(X_test)
    
    Reference:
        Keshavarzian et al., "RASTER: Random Segmented Threshold Extraction 
        for ROCKET", IEEE MLSP 2023.
    """
    
    def __init__(
        self,
        num_features: int = 10_000,
        max_dilations_per_kernel: int = 32,
        num_segments: int = 4,
        fixed_segments: bool = False,
        random_state: Optional[int] = None
    ):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.num_segments = num_segments
        self.fixed_segments = fixed_segments
        self.random_state = random_state
        self.parameters_ = None
        self.n_features_out_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "RASTER":
        """
        Fit the RASTER transform.
        
        Args:
            X: Time series data, shape (n_samples, series_length).
            y: Ignored, present for sklearn compatibility.
        
        Returns:
            self: The fitted transformer.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = self._validate_input(X)
        
        # Fit returns: dilations, num_features_per_dilation, random_size, biases
        self.parameters_ = _minirocket.fit(
            X,
            num_features=self.num_features,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            sizes=self.num_segments
        )
        
        # If fixed segments, override random sizes
        if self.fixed_segments:
            dilations, num_features_per_dilation, random_size, biases = self.parameters_
            random_size = np.ones(len(random_size), dtype=np.int64) * self.num_segments
            self.parameters_ = (dilations, num_features_per_dilation, random_size, biases)
        
        dilations, num_features_per_dilation, _, _ = self.parameters_
        self.n_features_out_ = 84 * np.sum(num_features_per_dilation)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform time series to RASTER features.
        
        Args:
            X: Time series data, shape (n_samples, series_length).
        
        Returns:
            Features array, shape (n_samples, n_features_out_).
        """
        if self.parameters_ is None:
            raise RuntimeError("RASTER must be fitted before transform")
        
        X = self._validate_input(X)
        return _minirocket.transform_refined(X, self.parameters_, 'ter')
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and convert input to correct dtype."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        return X.astype(np.float32)
    
    def __repr__(self) -> str:
        return f"RASTER(num_features={self.num_features}, num_segments={self.num_segments})"


class RASTERMultivariate:
    """
    RASTER for multivariate time series.
    
    Extends RASTER to handle multiple channels/variables.
    
    Parameters:
        num_features: Number of features to generate. Default 10,000.
        max_dilations_per_kernel: Maximum dilations per kernel. Default 32.
        random_state: Random seed for reproducibility. Default None.
    
    Example:
        >>> # X has shape (n_samples, n_channels, series_length)
        >>> raster_mv = RASTERMultivariate(num_features=10000)
        >>> raster_mv.fit(X_train)
        >>> X_train_features = raster_mv.transform(X_train)
    """
    
    def __init__(
        self,
        num_features: int = 10_000,
        max_dilations_per_kernel: int = 32,
        random_state: Optional[int] = None
    ):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state
        self.parameters_ = None
        self.n_features_out_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "RASTERMultivariate":
        """
        Fit the multivariate RASTER transform.
        
        Args:
            X: Time series data, shape (n_samples, n_channels, series_length).
            y: Ignored, present for sklearn compatibility.
        
        Returns:
            self: The fitted transformer.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = self._validate_input(X)
        
        self.parameters_ = _raster_mv.fit(
            X,
            num_features=self.num_features,
            max_dilations_per_kernel=self.max_dilations_per_kernel
        )
        
        # Compute output features
        _, _, dilations, num_features_per_dilation, _, _ = self.parameters_
        self.n_features_out_ = 84 * np.sum(num_features_per_dilation)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform multivariate time series to RASTER features.
        
        Args:
            X: Time series data, shape (n_samples, n_channels, series_length).
        
        Returns:
            Features array, shape (n_samples, n_features_out_).
        """
        if self.parameters_ is None:
            raise RuntimeError("RASTERMultivariate must be fitted before transform")
        
        X = self._validate_input(X)
        return _raster_mv.transform(X, self.parameters_)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and convert input to correct dtype."""
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (n_samples, n_channels, series_length), got {X.ndim}D")
        return X.astype(np.float32)
    
    def __repr__(self) -> str:
        return f"RASTERMultivariate(num_features={self.num_features})"


# Convenience function to get the right transformer
def get_transformer(
    method: str = "minirocket",
    multivariate: bool = False,
    **kwargs
) -> Union[ROCKET, MiniROCKET, RASTER, MiniROCKETMultivariate, RASTERMultivariate]:
    """
    Factory function to get a ROCKET family transformer.
    
    Args:
        method: One of 'rocket', 'minirocket', or 'raster'.
        multivariate: Whether to use multivariate version.
        **kwargs: Additional arguments passed to the transformer.
    
    Returns:
        Transformer instance.
    
    Example:
        >>> transformer = get_transformer('minirocket', num_features=5000)
        >>> transformer.fit(X_train)
        >>> features = transformer.transform(X_test)
    """
    method = method.lower()
    
    if method == "rocket":
        if multivariate:
            raise ValueError("ROCKET does not have a multivariate implementation")
        return ROCKET(**kwargs)
    
    elif method == "minirocket":
        if multivariate:
            return MiniROCKETMultivariate(**kwargs)
        return MiniROCKET(**kwargs)
    
    elif method == "raster":
        if multivariate:
            return RASTERMultivariate(**kwargs)
        return RASTER(**kwargs)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: rocket, minirocket, raster")

