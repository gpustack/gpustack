"""`{{name}}` substitution.

The reason this is a module rather than a backend method: substitution used to
live inside `InferenceBackend.replace_command_param`, reached only when a
version config has a `run_command` and no `built_in_frameworks` — which is the
definition of a non-built-in backend. Built-in vLLM and SGLang therefore never
rendered anything, and a connector variable reached the engine verbatim
(measured: `ZMQError: No such device (addr='tcp://{{worker_ip}}:5600')`).
"""

import logging

import pytest

from gpustack.utils.template import (
    deployment_variables,
    placeholders,
    render,
    render_values,
)


# --- the renderer -------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("tcp://{{worker_ip}}:5600", "tcp://10.0.0.7:5600"),
        ("{{worker_ip}}", "10.0.0.7"),
        ("{{worker_ip}}{{worker_ip}}", "10.0.0.710.0.0.7"),
        ("no placeholder", "no placeholder"),
        ("", ""),
    ],
)
def test_render_substitutes(text, expected):
    assert render(text, {"worker_ip": "10.0.0.7"}) == expected


def test_render_passes_none_through():
    assert render(None, {"a": "b"}) is None


def test_dotted_names_are_one_key():
    # Structure is the caller's business; the renderer only substitutes.
    variables = {
        "ports.kv_side_channel": 41100,
        "ports.kv_side_channel.count": 8,
        "roles.prefill.tensor_parallel_size": 8,
    }
    text = "{{ports.kv_side_channel}}+{{ports.kv_side_channel.count}}/{{roles.prefill.tensor_parallel_size}}"
    assert render(text, variables) == "41100+8/8"


@pytest.mark.parametrize("text", ["{{ worker_ip }}", "{{worker_ip", "{{1abc}}", "{{}}"])
def test_malformed_placeholders_are_not_placeholders(text):
    # Whitespace inside the braces in particular: a value that failed to
    # render has to look different from one that rendered, or it becomes the
    # literal that reached the engine.
    assert render(text, {"worker_ip": "10.0.0.7", "1abc": "x"}) == text


def test_an_unknown_name_is_left_alone_and_warned(caplog):
    with caplog.at_level(logging.WARNING):
        out = render("{{ports.kv}}", {}, context="env VLLM_X")
    assert out == "{{ports.kv}}"
    assert "{{ports.kv}}" in caplog.text
    assert "env VLLM_X" in caplog.text


def test_a_known_name_warns_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        render("{{a}}", {"a": 1})
    assert caplog.text == ""


def test_repeated_unknown_names_are_reported_once(caplog):
    with caplog.at_level(logging.WARNING):
        render("{{x}} {{x}} {{y}}", {})
    assert caplog.text.count("{{x}}") == 1
    assert "{{y}}" in caplog.text


def test_a_none_value_renders_empty():
    assert render("[{{a}}]", {"a": None}) == "[]"


def test_placeholders_lists_names_in_order():
    assert placeholders("{{b}} {{a}} {{b}}") == ["b", "a", "b"]
    assert placeholders(None) == []


# --- env values ---------------------------------------------------------


def test_render_values_renders_values_not_keys():
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "{{worker_ip}}",
        "{{worker_ip}}": "literal-key",
    }
    out = render_values(env, {"worker_ip": "10.0.0.7"})

    assert out["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "10.0.0.7"
    assert out["{{worker_ip}}"] == "literal-key"


def test_render_values_passes_empty_through():
    assert render_values(None, {"a": 1}) is None
    assert render_values({}, {"a": 1}) == {}


def test_env_values_do_not_see_each_other():
    # Deliberate: cross-references would make the result depend on iteration
    # order and admit cycles.
    out = render_values({"A": "1", "B": "{{A}}"}, {})
    assert out["B"] == "{{A}}"


# --- the variable set ---------------------------------------------------


def test_deployment_variables_matches_the_original_command_semantics():
    # The six original names always render, and an absent one renders empty.
    # That is what replace_command_param has always done, and the run-command
    # path must not change behaviour.
    variables = deployment_variables()
    assert variables["model_path"] == ""
    assert variables["worker_ip"] == ""
    assert variables["model_name"] == ""
    assert variables["gpu_count"] == ""
    assert variables["gpu_ids"] == ""


def test_a_portless_instance_renders_empty_rather_than_the_word_none():
    # A missing port renders empty rather than `str(port)`'s literal "None".
    # Both values are broken for a command that wants a port; "" is at least
    # not a plausible-looking token, and nothing can reasonably depend on
    # "None".
    assert render("{{port}}", deployment_variables()) == ""
    assert render("{{port}}", deployment_variables(port=8000)) == "8000"


def test_deployment_variables_formats_gpu_ids_as_a_csv():
    variables = deployment_variables(gpu_ids=[2, 0, 1])
    assert variables["gpu_ids"] == "2,0,1"
    assert deployment_variables(gpu_ids=[])["gpu_ids"] == ""


def test_role_and_group_are_absent_without_a_role(caplog):
    # Omitted rather than empty, so a non-PD deployment referencing them gets
    # the unresolved warning instead of a silent empty string.
    variables = deployment_variables()
    assert "role" not in variables
    assert "group_id" not in variables

    with caplog.at_level(logging.WARNING):
        assert render("{{role}}", variables) == "{{role}}"
    assert "{{role}}" in caplog.text


def test_role_and_group_render_for_a_pd_instance():
    variables = deployment_variables(role="prefill", group_id="g-1")
    assert render("{{role}}/{{group_id}}", variables) == "prefill/g-1"


def test_the_measured_failure_now_renders():
    # The side-channel host must reach the engine as an address; an unrendered
    # placeholder gets there as a literal and raises ZMQError.
    variables = deployment_variables(worker_ip="10.0.0.7")
    out = render_values({"VLLM_NIXL_SIDE_CHANNEL_HOST": "{{worker_ip}}"}, variables)
    assert out["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "10.0.0.7"


def test_a_missing_variable_source_is_not_fatal(caplog):
    # Rendering enriches an env value; it must not gain the power to end a
    # start. `InferenceServer._template_variables` reads every source
    # tolerantly for this reason: a caller holding only `_model` must not
    # raise from underneath `_get_configured_env`.
    with caplog.at_level(logging.WARNING):
        out = render_values({"A": "{{worker_ip}}:{{port}}"}, deployment_variables())
    assert out["A"] == ":"
    assert caplog.text == ""
