"""The packaged chart's location has to agree with how the server serves it.

Two facts have to line up, and they live in different files: ``hack/install.sh``
writes the chart into the UI's static tree, and ``gpustack/routes/ui.py`` mounts
that tree at ``/static``. Getting either wrong produces a manifest whose Job
fetches a 404 — a failure that only shows up inside a cluster, minutes after the
mistake.
"""

import pathlib

from gpustack.k8s.chart import (
    CHART_STATIC_PATH,
    chart_asset_path,
    chart_download_url,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_asset_path_is_under_the_mounted_static_tree():
    ui_static = REPO_ROOT / "gpustack" / "ui" / "static"
    # `relative_to` raises if the asset escapes the mounted directory, which is
    # the failure mode worth catching: a path that resolves but is not served.
    assert chart_asset_path().relative_to(ui_static)


def test_static_path_matches_the_url_the_mount_serves():
    # routes/ui.py mounts gpustack/ui/{css,js,static} at /{css,js,static}, so a
    # path under the static tree is reachable at exactly that prefix.
    assert CHART_STATIC_PATH.startswith("static/")
    assert chart_download_url("http://gpustack:80").endswith(f"/{CHART_STATIC_PATH}")


def test_install_script_writes_where_the_asset_is_expected():
    script = (REPO_ROOT / "hack" / "install.sh").read_text()
    target = str(pathlib.PurePosixPath(CHART_STATIC_PATH).parent)
    assert f"gpustack/ui/{target}" in script, (
        f"hack/install.sh does not write the chart to gpustack/ui/{target}; "
        "the packaged chart would not be served"
    )
    # Ordering matters as much as the path: download_ui removes the whole ui
    # directory, so packaging has to come after it.
    assert script.index("download_ui\n") < script.index("package_chart\n")


def test_the_windows_build_packages_the_chart_too():
    """Both dependency paths have to produce it, or one wheel serves 404s.

    The Windows build runs its own script rather than `install.sh`, so a chart
    packaged only in the Unix path yields a wheel that installs cleanly and then
    refuses every manifest request — `chart_available()` finds nothing, and the
    failure surfaces in a registration, not in a build.
    """
    script = (REPO_ROOT / "hack" / "windows" / "install.ps1").read_text()
    target = str(pathlib.PurePosixPath(CHART_STATIC_PATH).parent)
    assert (
        f"gpustack/ui/{target}" in script
    ), f"hack/windows/install.ps1 does not write the chart to gpustack/ui/{target}"
    # Same ordering constraint as the Unix path: Get-UI removes that directory.
    assert script.index("Get-UI\n") < script.index("Package-Chart\n")


def test_download_url_tolerates_a_trailing_slash():
    # `server_url` reaches this from cluster registration, where it is whatever
    # the caller configured; a doubled slash would still resolve but makes the
    # rendered manifest look broken.
    assert chart_download_url("http://gpustack:80/") == chart_download_url(
        "http://gpustack:80"
    )
