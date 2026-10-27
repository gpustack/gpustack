from datetime import datetime

import pytest
from pydantic import ValidationError

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.benchmarks import (
    MAX_BENCHMARK_RESULT_POINTS,
    _result_to_public,
    get_benchmark_results,
    _validate_load_config,
    apply_progress_invariant,
    order_benchmark_export_fields,
)
from gpustack.schemas.benchmark import (
    DATASET_SEED_MAX,
    DATASET_SEED_MIN,
    SLO_THRESHOLDS,
    Benchmark,
    BenchmarkCreate,
    BenchmarkFullPublic,
    BenchmarkLoadModeEnum,
    BenchmarkPublic,
    BenchmarkResultCreate,
    BenchmarkResultPublic,
    BenchmarkStateEnum,
    BenchmarkStateUpdate,
    benchmark_load_axis,
    benchmark_load_mode,
    generate_dataset_seed,
)


def test_generate_dataset_seed_leaves_room_for_per_stage_increment():
    # A multi-stage run derives each stage's seed as base + stage index, so the
    # base must stay far enough below the generator's 2**32 seed ceiling that the
    # last stage of even a long ramp can't overflow it.
    seeds = {generate_dataset_seed() for _ in range(50)}

    assert all(DATASET_SEED_MIN <= s <= DATASET_SEED_MAX for s in seeds)
    assert DATASET_SEED_MAX + 10_000 < 2**32
    # Two benchmarks of the same config must not share prompts by construction.
    assert len(seeds) > 1


def test_order_benchmark_export_fields_puts_identifying_fields_first():
    benchmark = {
        "description": "benchmark description",
        "dataset_output_tokens": 256,
        "snapshot": {},
        "name": "benchmark-a",
        "request_rate": 10,
        "model_name": "model-a",
        "profile": "Custom",
        "dataset_name": "Random",
        "model_instance_name": "model-a-1",
        "total_requests": 100,
        "dataset_input_tokens": 128,
        "dataset_seed": 42,
    }

    ordered = order_benchmark_export_fields(benchmark)

    assert list(ordered) == [
        "name",
        "model_name",
        "model_instance_name",
        "profile",
        "dataset_name",
        "request_rate",
        "total_requests",
        "dataset_input_tokens",
        "dataset_output_tokens",
        "dataset_seed",
        "description",
        "snapshot",
    ]


def _create(**fields) -> BenchmarkCreate:
    """A minimal valid BenchmarkCreate, overridden per test."""
    return BenchmarkCreate(
        name="bm",
        model_instance_name="mi",
        dataset_name="Random",
        dataset_input_tokens=128,
        dataset_output_tokens=128,
        **fields,
    )


class TestLoadTypeIsAnEnum:
    """A typo in `load_type` used to fall through to the rate axis, so a run
    labelled "concurrency" silently executed as an open-loop rate sweep. It is now
    rejected at the API boundary instead."""

    def test_a_wrong_case_load_type_is_refused(self):
        with pytest.raises(ValidationError):
            _create(load_type="Concurrency")

    def test_the_two_valid_axes_survive_as_their_wire_values(self):
        # Stored and forwarded as the VALUE, not the enum member name: the column is
        # an AutoString, and the runner's --axis / the UI both read these strings.
        for value in ("fixed_rate", "concurrency"):
            assert _create(load_type=value).load_type == value

    def test_the_axis_name_the_runner_wants_is_derived_not_repeated(self):
        # The column says "fixed_rate" while the runner's flag says "rate", so the
        # two are not interchangeable — hence one place that maps between them.
        assert benchmark_load_axis(_create(load_type="concurrency")) == "concurrency"
        assert benchmark_load_axis(_create(load_type="fixed_rate")) == "rate"
        assert benchmark_load_axis(_create()) == "rate"  # unset => rate axis


class TestLoadModeIsExplicit:
    """auto_tune / stages / single is one decision, read from one function: the
    command builder, the result collection and the ready-file count all used to
    re-derive this precedence with their own if/elif chain."""

    def test_each_shape_is_named(self):
        assert (
            benchmark_load_mode(_create(auto_tune=True))
            is BenchmarkLoadModeEnum.AUTO_TUNE
        )
        assert (
            benchmark_load_mode(_create(stages=[{"rate": 2}]))
            is BenchmarkLoadModeEnum.STAGES
        )
        assert benchmark_load_mode(_create()) is BenchmarkLoadModeEnum.SINGLE

    def test_an_empty_stage_list_is_not_the_stages_shape(self):
        # Falsy, so it must not select a shape that would then run zero stages.
        assert benchmark_load_mode(_create(stages=[])) is BenchmarkLoadModeEnum.SINGLE


class TestValidateLoadConfig:
    def test_auto_tune_and_stages_together_are_refused(self):
        # Both set, the runner silently honours auto_tune and drops the stages the
        # user configured — so the combination is rejected rather than ranked.
        with pytest.raises(BadRequestException) as e:
            _validate_load_config(_create(auto_tune=True, stages=[{"rate": 2}]))
        assert "mutually exclusive" in str(e.value.message)

    def test_an_inverted_search_range_is_refused(self):
        with pytest.raises(BadRequestException) as e:
            _validate_load_config(
                _create(auto_tune=True, lower_bound=64, upper_bound=8)
            )
        assert "less than upper_bound" in str(e.value.message)

    def test_an_empty_search_range_is_refused(self):
        # lower == upper leaves nothing to search.
        with pytest.raises(BadRequestException):
            _validate_load_config(_create(auto_tune=True, lower_bound=8, upper_bound=8))

    def test_a_valid_range_passes(self):
        _validate_load_config(_create(auto_tune=True, lower_bound=4, upper_bound=1024))

    @pytest.mark.parametrize(
        "field", ["max_points", "max_total_seconds", "max_seconds", "turns"]
    )
    def test_non_positive_budgets_are_refused(self, field):
        with pytest.raises(BadRequestException) as e:
            _validate_load_config(_create(**{field: 0}))
        assert field in str(e.value.message)

    def test_a_point_budget_the_grid_cannot_hold_is_refused(self):
        # The results upload caps a grid at MAX_BENCHMARK_RESULT_POINTS, and that
        # check runs at the END of the run: without this one, a 600-point benchmark
        # ramps for its whole time budget and then has the entire curve rejected by
        # the terminal sync, leaving a state_message and nothing else.
        with pytest.raises(BadRequestException) as e:
            _validate_load_config(
                _create(auto_tune=True, max_points=MAX_BENCHMARK_RESULT_POINTS + 1)
            )
        assert "max_points" in str(e.value.message)
        assert str(MAX_BENCHMARK_RESULT_POINTS) in str(e.value.message)

    def test_the_ceiling_itself_is_allowed(self):
        _validate_load_config(
            _create(auto_tune=True, max_points=MAX_BENCHMARK_RESULT_POINTS)
        )

    def test_more_manual_stages_than_the_grid_can_hold_is_refused(self):
        # The other way to ask for more rows than the table accepts.
        stages = [{"rate": 1}] * (MAX_BENCHMARK_RESULT_POINTS + 1)
        with pytest.raises(BadRequestException) as e:
            _validate_load_config(_create(stages=stages))
        assert "stages" in str(e.value.message)

    @pytest.mark.parametrize("t", SLO_THRESHOLDS, ids=lambda t: t.attr)
    def test_every_slo_threshold_must_be_positive(self, t):
        # Parametrized off the single source of truth, so a threshold added there is
        # covered here without editing this test.
        with pytest.raises(BadRequestException) as e:
            _validate_load_config(_create(**{t.attr: 0}))
        assert t.attr in str(e.value.message)

    def test_a_stage_without_a_numeric_rate_is_refused(self):
        for stages in ([{}], [{"rate": "fast"}], [{"rate": None}], [{"rate": True}]):
            with pytest.raises(BadRequestException):
                _validate_load_config(_create(stages=stages))

    def test_a_stage_that_is_not_an_object_never_reaches_the_check(self):
        # `stages` is List[Dict[str, Any]], so the item TYPE is pydantic's contract
        # and comes back as a 422 at the boundary — _validate_load_config only has
        # to judge the contents.
        with pytest.raises(ValidationError):
            _create(stages=["nope"])

    def test_a_non_positive_stage_rate_is_refused(self):
        with pytest.raises(BadRequestException):
            _validate_load_config(_create(stages=[{"rate": 2}, {"rate": 0}]))

    def test_valid_stages_pass(self):
        _validate_load_config(
            _create(stages=[{"rate": 2}, {"rate": 4, "max_seconds": 60}])
        )

    @pytest.mark.parametrize("key", ["max_requests", "max_seconds"])
    def test_an_unusable_per_stage_constraint_is_refused(self, key):
        # These ride into `--stages` as JSON verbatim, so a 0 / negative / non-numeric
        # one is not caught until the container acts on it minutes later.
        for bad in (0, -1, "soon"):
            with pytest.raises(BadRequestException) as e:
                _validate_load_config(_create(stages=[{"rate": 2, key: bad}]))
            assert key in str(e.value.message)

    def test_a_stage_constraint_left_out_is_fine(self):
        # Absent means "no per-stage cap", which is the common case.
        _validate_load_config(
            _create(stages=[{"rate": 2, "max_requests": None, "max_seconds": None}])
        )

    @pytest.mark.parametrize("field", ["warmup", "cooldown", "max_errors"])
    def test_a_negative_pass_through_knob_is_refused(self, field):
        with pytest.raises(BadRequestException) as e:
            _validate_load_config(_create(**{field: -1}))
        assert field in str(e.value.message)

    @pytest.mark.parametrize("field", ["warmup", "cooldown", "max_errors"])
    def test_zero_stays_a_legal_setting(self, field):
        # Unlike the "<= N" budgets, 0 is meaningful here: no warmup, no cooldown,
        # tolerate no errors at all. It must not be swept up with the negatives.
        _validate_load_config(_create(**{field: 0}))

    def test_an_inverted_token_window_is_refused(self):
        with pytest.raises(BadRequestException) as e:
            _validate_load_config(_create(dataset_input_min=512, dataset_input_max=128))
        assert "dataset_input_min" in str(e.value.message)

    def test_a_valid_token_window_passes(self):
        _validate_load_config(
            _create(
                dataset_input_min=64,
                dataset_input_max=256,
                dataset_output_min=8,
                dataset_output_max=8,
            )
        )


class TestProgressInvariant:
    """Progress is monotonic within a run, but a starting run is not bound by the
    previous one's value."""

    def _row(self, state, progress):
        from types import SimpleNamespace

        return SimpleNamespace(state=state, progress=progress)

    def test_a_backward_report_is_held_at_the_floor(self):
        # A later stage reporting its own early percentage must not rewind the bar.
        update = BenchmarkStateUpdate(progress=40.0)
        apply_progress_invariant(update, self._row(BenchmarkStateEnum.RUNNING, 75.0))
        assert update.progress == 75.0

    def test_forward_progress_passes_through(self):
        update = BenchmarkStateUpdate(progress=80.0)
        apply_progress_invariant(update, self._row(BenchmarkStateEnum.RUNNING, 75.0))
        assert update.progress == 80.0

    def test_entering_running_resets_a_stale_progress(self):
        # A re-run starts over: the previous run's 100% is not this run's floor.
        update = BenchmarkStateUpdate(state=BenchmarkStateEnum.RUNNING)
        apply_progress_invariant(update, self._row(BenchmarkStateEnum.QUEUED, 100.0))
        assert update.progress == 0.0
        assert "progress" in update.model_fields_set

    def test_entering_running_keeps_a_progress_the_caller_stated(self):
        # The patch carries both: that is a report of where the run IS, so zeroing it
        # would throw away a real measurement.
        update = BenchmarkStateUpdate(state=BenchmarkStateEnum.RUNNING, progress=12.0)
        apply_progress_invariant(update, self._row(BenchmarkStateEnum.QUEUED, 100.0))
        assert update.progress == 12.0

    def test_a_patch_without_progress_is_left_alone(self):
        update = BenchmarkStateUpdate(state=BenchmarkStateEnum.COMPLETED)
        apply_progress_invariant(update, self._row(BenchmarkStateEnum.RUNNING, 75.0))
        assert update.progress is None
        assert "progress" not in update.model_fields_set


class TestBenchmarkResultCreate:
    """The upload contract carries the measurement and nothing else."""

    @pytest.mark.parametrize(
        "field", ["id", "benchmark_id", "created_at", "updated_at", "deleted_at"]
    )
    def test_server_owned_fields_are_not_settable(self, field):
        # An untyped dict body let a client pin a primary key (a 500 on collision) or
        # set deleted_at on the row it had just written.
        assert field not in BenchmarkResultCreate.model_fields
        row = BenchmarkResultCreate.model_validate({"rate": 4.0, field: 1})
        assert not hasattr(row, field) or getattr(row, field) is None

    def test_the_measurement_itself_round_trips(self):
        row = BenchmarkResultCreate.model_validate(
            {
                "rate": 8.0,
                "sequence": 3,
                "strategy_type": "concurrent",
                "input_tokens": 128,
                "tokens_per_second_mean": 9100.0,
                "raw_metrics": {"a": 1},
            }
        )
        assert row.rate == 8.0
        assert row.sequence == 3
        assert row.tokens_per_second_mean == 9100.0
        assert row.raw_metrics == {"a": 1}

    def test_a_non_numeric_measurement_is_a_validation_error_not_a_500(self):
        with pytest.raises(ValidationError):
            BenchmarkResultCreate.model_validate({"rate": "fast"})


class TestResultResponseAvoidsDeferredLoad:
    """`raw_metrics` is deferred unless asked for, so the response must be built
    without touching it: reading a deferred column is a lazy load, and on an async
    session that raises rather than quietly fetching."""

    class _LoadedRow:
        """A row whose every column was loaded."""

        raw_metrics = {"a": 1}

        def __init__(self):
            for name in BenchmarkResultPublic.model_fields:
                if name != "raw_metrics":
                    setattr(self, name, None)
            self.id = 1
            self.benchmark_id = 7
            self.created_at = self.updated_at = datetime(2026, 1, 1)
            self.rate = 8.0
            self.sequence = 2
            self.tokens_per_second_mean = 9100.0

    class _DeferredRow(_LoadedRow):
        """Same row, but loaded with defer(raw_metrics): touching that attribute is
        the lazy load we must avoid, so here it blows up instead."""

        @property
        def raw_metrics(self):  # pragma: no cover - must never be reached
            raise AssertionError("a deferred column was read")

    def test_the_summary_never_reads_the_deferred_column(self):
        out = _result_to_public(self._DeferredRow(), include_raw=False)
        assert out.raw_metrics is None
        # ...while the measurement itself still comes through.
        assert out.rate == 8.0
        assert out.sequence == 2
        assert out.tokens_per_second_mean == 9100.0

    def test_asking_for_the_raw_dump_reads_it(self):
        out = _result_to_public(self._LoadedRow(), include_raw=True)
        assert out.raw_metrics == {"a": 1}

    def test_every_response_field_is_populated(self):
        # Built field by field, so a field added to the model must not be dropped.
        out = _result_to_public(self._DeferredRow(), include_raw=False)
        assert set(out.model_dump()) == set(BenchmarkResultPublic.model_fields)

    def test_raw_metrics_are_returned_by_default(self):
        # The detail page reads its percentile charts, per-point duration and
        # early-stop reason out of raw_metrics, so an opt-IN default would silently
        # empty half of it. The flag exists for callers that only want the curve.
        import inspect

        default = inspect.signature(get_benchmark_results).parameters["include_raw"]
        assert default.default.default is True


class TestValidateDatasetSeed:
    """A client-pinned seed must fit the range the generator reserves — the bound
    exists because a multi-stage run offsets the base per stage, so a base near
    numpy's 2**32 ceiling overflows on the last stages."""

    def test_a_seed_past_the_reserved_headroom_is_refused(self):
        with pytest.raises(BadRequestException) as e:
            _validate_load_config(_create(dataset_seed=2**32 - 1))
        assert "dataset_seed" in str(e.value.message)

    def test_a_negative_seed_is_refused(self):
        # numpy rejects a negative seed outright.
        with pytest.raises(BadRequestException):
            _validate_load_config(_create(dataset_seed=-1))

    def test_zero_is_refused(self):
        # The generated range starts at 1, so 0 is outside what the server itself
        # would ever pick.
        with pytest.raises(BadRequestException):
            _validate_load_config(_create(dataset_seed=0))

    @pytest.mark.parametrize("seed", [DATASET_SEED_MIN, 42, DATASET_SEED_MAX])
    def test_seeds_inside_the_range_pass(self, seed):
        _validate_load_config(_create(dataset_seed=seed))

    def test_a_generated_seed_always_passes_its_own_check(self):
        # The generator and the validator must agree, or the server could produce a
        # benchmark its own API would reject on re-submit (e.g. a clone).
        for _ in range(50):
            _validate_load_config(_create(dataset_seed=generate_dataset_seed()))

    def test_a_non_random_dataset_is_not_seed_checked(self):
        # Only the synthetic dataset generates prompts from the seed; a file dataset
        # carries whatever the row happened to have.
        _validate_load_config(
            BenchmarkCreate(
                name="bm",
                model_instance_name="mi",
                dataset_name="ShareGPT",
                dataset_seed=2**40,
            )
        )

    def test_no_seed_is_fine(self):
        # The server fills it in at creation.
        _validate_load_config(_create())


class TestCreateCannotFabricateAConclusion:
    """`validate_and_mutate_benchmark_in` builds the row with
    `Benchmark(**benchmark_in.model_dump())` and then overrides only what it derives,
    so every other field on the create schema lands in the row exactly as sent. The
    run's own account of itself — progress, outcome, conclusion — is therefore kept
    off `BenchmarkBase` (see `BenchmarkRuntime`), the same treatment
    `BenchmarkResultCreate` gets for the per-point upload."""

    RUNTIME_FIELDS = (
        "state",
        "state_message",
        "progress",
        "pid",
        "peak_rate",
        "slo_met_rate",
        "recommended_rate",
        "validity",
    )

    @pytest.mark.parametrize("field", RUNTIME_FIELDS)
    def test_the_create_body_does_not_carry_it(self, field):
        assert field not in BenchmarkCreate.model_fields

    def test_a_forged_conclusion_does_not_reach_the_row(self):
        # The shape that motivated this: a POST that arrives already "finished".
        forged = _create(
            state=BenchmarkStateEnum.COMPLETED,
            progress=100,
            peak_rate=9999,
            slo_met_rate=9999,
            recommended_rate=9999,
            validity={"sufficient": True, "warnings": []},
        )
        assert not set(forged.model_dump()) & set(self.RUNTIME_FIELDS)
        row = Benchmark(**forged.model_dump())
        assert row.state is BenchmarkStateEnum.PENDING
        assert (row.peak_rate, row.slo_met_rate, row.recommended_rate) == (
            None,
            None,
            None,
        )
        assert row.validity is None
        assert row.progress is None

    @pytest.mark.parametrize("field", RUNTIME_FIELDS)
    def test_readers_still_see_it(self, field):
        # Narrowing the write side must not narrow the read side: the list page reads
        # state and the coverage verdict, the detail page reads the best points.
        assert field in Benchmark.model_fields
        assert field in BenchmarkPublic.model_fields
        assert field in BenchmarkFullPublic.model_fields

    @pytest.mark.parametrize("field", RUNTIME_FIELDS)
    def test_the_column_still_exists(self, field):
        # The mixin moved where the field is DECLARED, not whether it is stored.
        assert field in Benchmark.__table__.columns

    def test_the_worker_can_still_write_them(self):
        # PATCH /{id}/state is the one door in, and it stays open.
        for field in ("state", "state_message", "progress", "pid"):
            assert field in BenchmarkStateUpdate.model_fields
        for field in ("peak_rate", "slo_met_rate", "recommended_rate", "validity"):
            assert field in BenchmarkStateUpdate.model_fields

    def test_the_config_fields_are_untouched(self):
        # Sanity guard on the split: what the client legitimately configures has to
        # stay on the create body.
        for field in ("auto_tune", "lower_bound", "max_points", "slo_avg_ttft_ms"):
            assert field in BenchmarkCreate.model_fields
