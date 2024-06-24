import inflection


def to_snake_case(value):
    return inflection.underscore(value)


def to_plural(value):
    return inflection.pluralize(value)


def to_underscore_plural(value):
    return inflection.pluralize(inflection.underscore(value))


def to_dash_plural(value):
    return inflection.pluralize(inflection.dasherize(inflection.underscore(value)))
