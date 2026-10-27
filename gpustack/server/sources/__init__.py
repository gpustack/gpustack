"""Shared, reusable source layer.

A *source* carries content (inline for FILE, or a URL fetched at PUT time for
URL) that a leader reconcile materializes into a derived table. Three consumers
share it: ``InferenceRunnerSource`` (runner overrides), ``CatalogSource`` (model
catalog) and ``InferenceBackendSource`` (community backends).

Layout: ``core`` the shared behaviour (fetch, order, materialize), ``routes`` the
HTTP config API, ``probe`` the official-content fetch. What a source *is* lives in
``gpustack.schemas.source``, so the dependency runs one way — no ``schemas`` module
imports this package.

Import from the submodules directly; nothing is re-exported here, since ``routes``
would drag the auth/client stack in for anyone who only wanted ``core``.
"""
