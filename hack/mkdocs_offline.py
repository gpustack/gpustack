"""Keep the bundled error page's resources under the server's help mount."""

import re

from mkdocs.plugins import event_priority


def help_root(config):
    """Locate the current language inside the combined bundled site."""
    language = config["theme"].get("language", "en")
    return "/help/" + (f"{language}/" if language != "en" else "")


def on_template_context(context, *, template_name, config):
    if template_name == "404.html":
        context["base_url"] = help_root(config)
    return context


@event_priority(-100)
def on_post_template(output, *, template_name, config):
    if template_name == "404.html":
        # Privacy runs at -50 and uses site_url to locate vendored assets.
        # With no public site_url, those URLs still point at the server root.
        output = re.sub(
            r"(\b(?:href|src)=[\"']?)/(?!/|help(?:/|[\"' >]))",
            lambda match: match[1] + help_root(config),
            output,
        )
    return output
