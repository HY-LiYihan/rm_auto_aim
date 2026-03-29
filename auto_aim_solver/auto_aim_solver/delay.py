from dataclasses import dataclass


@dataclass
class DelayConfig:
    pipeline_delay_s: float = 0.02
    control_delay_s: float = 0.015


def compute_data_age_s(now_sec: float, sample_sec: float) -> float:
    return max(0.0, now_sec - sample_sec)


def compute_delta_t_s(
    data_age_s: float,
    flight_time_s: float,
    config: DelayConfig,
) -> float:
    return max(0.0, data_age_s + config.pipeline_delay_s + config.control_delay_s + flight_time_s)
