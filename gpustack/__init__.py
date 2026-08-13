__version__ = '0.0.0'
__git_commit__ = 'HEAD'
# Requires the auto-tune/stages CLI (--auto-tune, --axis, --stages, --sla-*,
# --seed-increment, --detect-saturation) and the guidellm 0.7.1 backend that takes
# path-keyed `request_handlers`. v0.0.4 has neither, and the backend kwarg is sent
# on EVERY run, so pointing at an older image breaks single-rate runs too.
__benchmark_runner_version__ = 'v0.0.5'
__operator_version__ = 'v0.8.2'
