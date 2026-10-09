from dataclasses import asdict, dataclass, fields
import math


def finite(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f'{name} outside allowed range')


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class Config:
    mode: str
    expected_candidate_count: int
    training_candidate_count: int
    sigma: float
    candidate_cluster_threshold: float
    active_user_clusters: int
    index_policy: str
    index_top_clusters: int | None
    index_mass_threshold: float | None
    threshold_policy: str
    threshold_value: float | None
    activity_threshold_intercept: float | None
    activity_threshold_slope: float | None
    user_thresholds: dict
    neighbor_k_mode: str
    neighbor_k: int | None
    fixed_neighbor_k: int
    soft_context_k: int
    instruction: str
    response_prefix: str
    target_history_limit: int | None
    neighbor_history_limit: int
    likelihood_reduction: str
    candidate_terminator: str
    max_prompt_tokens: int
    max_sequence_tokens: int
    cold_start_cutoff: int
    gate_temperature: float
    gate_eta: float
    gate_initial_threshold: float
    schema_version: int = 1
    implementation: str = 'reference_reconstruction'

    def __post_init__(self):
        if self.schema_version != 1 or self.implementation != 'reference_reconstruction':
            raise ValueError('unsupported reconstruction schema')
        if self.mode not in {'hard_text_reference', 'soft_embedding_gate'}:
            raise ValueError('unknown mode')
        finite(self.sigma, 'sigma', 0)
        finite(self.gate_temperature, 'gate_temperature', 0)
        if self.sigma == 0 or self.gate_temperature == 0:
            raise ValueError('sigma and gate_temperature must be strictly positive')
        for name in ('candidate_cluster_threshold', 'gate_initial_threshold'):
            finite(getattr(self, name), name, 0, 1)
        if not 0 < self.gate_initial_threshold < 1:
            raise ValueError('gate_initial_threshold must be strictly between 0 and 1')
        finite(self.gate_eta, 'gate_eta', 0)
        for name in ('expected_candidate_count', 'training_candidate_count', 'active_user_clusters', 'fixed_neighbor_k', 'soft_context_k', 'neighbor_history_limit', 'max_prompt_tokens', 'max_sequence_tokens', 'cold_start_cutoff'):
            positive_int(getattr(self, name), name)
        if self.max_sequence_tokens <= self.max_prompt_tokens:
            raise ValueError('max_sequence_tokens must exceed max_prompt_tokens')
        for name in ('neighbor_k', 'target_history_limit', 'index_top_clusters'):
            if getattr(self, name) is not None:
                positive_int(getattr(self, name), name)
        if self.index_policy not in {'literal_positive', 'user_top_clusters', 'user_mass_threshold'}:
            raise ValueError('unknown index_policy')
        if (self.index_policy == 'user_top_clusters') != (self.index_top_clusters is not None):
            raise ValueError('index_top_clusters is required only for user_top_clusters')
        if (self.index_policy == 'user_mass_threshold') != (self.index_mass_threshold is not None):
            raise ValueError('index_mass_threshold is required only for user_mass_threshold')
        if self.index_mass_threshold is not None:
            finite(self.index_mass_threshold, 'index_mass_threshold', 0, 1)
        if self.threshold_policy not in {'fixed', 'activity_logistic', 'per_user'}:
            raise ValueError('unknown threshold_policy')
        if (self.threshold_policy == 'fixed') != (self.threshold_value is not None):
            raise ValueError('threshold_value is required only for fixed threshold')
        if self.threshold_value is not None:
            finite(self.threshold_value, 'threshold_value', 0, 1)
            if self.mode == 'soft_embedding_gate' and (not 0 < self.threshold_value < 1 or self.gate_initial_threshold != self.threshold_value):
                raise ValueError('soft fixed policy requires threshold_value strictly in (0,1) and identical gate_initial_threshold')
        if not isinstance(self.user_thresholds, dict):
            raise ValueError('user_thresholds must be an object')
        if (self.threshold_policy == 'per_user') != bool(self.user_thresholds):
            raise ValueError('user_thresholds must be nonempty only for per_user')
        for key, value in self.user_thresholds.items():
            if not isinstance(key, str) or not key:
                raise ValueError('threshold user IDs must be nonempty strings')
            finite(value, f'user_thresholds[{key}]', 0, 1)
        for name in ('activity_threshold_intercept', 'activity_threshold_slope'):
            if (self.threshold_policy == 'activity_logistic') != (getattr(self, name) is not None):
                raise ValueError(f'{name} is required only for activity_logistic')
            if getattr(self, name) is not None:
                finite(getattr(self, name), name)
        if self.neighbor_k_mode not in {'none', 'after_threshold', 'topk_override'}:
            raise ValueError('unknown neighbor_k_mode')
        if (self.neighbor_k_mode != 'none') != (self.neighbor_k is not None):
            raise ValueError('neighbor_k must be supplied exactly when its mode uses K')
        if self.likelihood_reduction not in {'sum', 'mean'}:
            raise ValueError('likelihood_reduction must be sum or mean')
        for name in ('instruction', 'response_prefix', 'candidate_terminator'):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f'{name} must be a string')
        if not self.instruction or not self.response_prefix:
            raise ValueError('instruction and response_prefix cannot be empty')

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError('config must be an object')
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f'unknown config fields: {sorted(unknown)}')
        try:
            return cls(**raw)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

    def to_dict(self):
        return asdict(self)
