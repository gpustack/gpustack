__version__ = '0.0.0'
__git_commit__ = 'HEAD'
# Requires the auto-tune/stages CLI (--auto-tune, --axis, --stages, --slo-*,
# --seed-increment, --detect-saturation) and the guidellm 0.7.1 backend that takes
# path-keyed `request_handlers`. v0.0.4 has neither, and the backend kwarg is sent
# on EVERY run, so pointing at an older image breaks single-rate runs too.
#
# Also needs --progress-ca-cert / --progress-insecure-skip-tls-verify, which the
# benchmark container is given when the server sits behind a private CA.
#
# v0.0.7 is the floor, not a preference: it renamed the latency-target flags
# --sla-* -> --slo-* and the `sla_failed` stop reason to `slo_failed`. An older
# image rejects every flag this server sends for an SLO run, and would report a
# stop reason the detail page has no text for.
__benchmark_runner_version__ = 'v0.0.7'
__operator_version__ = 'v0.8.6'
