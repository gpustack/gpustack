"""Run search when the query changes through paste or IME composition."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_search_input_notifies_material_of_committed_text():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the search input handler")
    root = Path(__file__).resolve().parents[2]
    script = """
const assert = require('node:assert/strict');
const listeners = new Map();
globalThis.document = {
  addEventListener: (type, listener) => listeners.set(type, listener)
};
require('./docs/javascripts/search-input.js');

const queries = [];
const target = new EventTarget();
target.matches = selector => selector === '[data-md-component="search-query"]';
target.addEventListener('keyup', () => queries.push(target.value));

target.value = 'ji';
listeners.get('input')({target, isComposing: true});
assert.deepEqual(queries, []);

target.value = '集群';
listeners.get('compositionend')({target});
assert.deepEqual(queries, ['集群']);

target.value = '集群管理';
listeners.get('input')({target, isComposing: false});
assert.deepEqual(queries, ['集群', '集群管理']);

listeners.get('input')({target: {matches: () => false}});
listeners.get('input')({target: new EventTarget()});
listeners.get('input')({target: null});
assert.deepEqual(queries, ['集群', '集群管理']);
"""
    subprocess.run([node, "-e", script], cwd=root, check=True, timeout=15)
