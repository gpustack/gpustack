import base64

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.schema import CreateTable

from gpustack.schemas.site_settings import (
    SiteSettings,
    SiteSettingsPublic,
    SiteSettingsUpdate,
)


def settings(**overrides):
    return SiteSettingsUpdate(
        **{'site_name': 'GPUStack', 'primary_color': '#007BFF', **overrides}
    )


@pytest.mark.parametrize(
    'field', ['logo', 'logo_dark', 'mini_logo', 'mini_logo_dark', 'favicon']
)
def test_raster_assets_are_bounded_and_typed(field):
    png = 'data:image/png;base64,' + base64.b64encode(b'\x89PNG\r\n\x1a\n').decode()
    assert getattr(settings(**{field: png}), field) == png
    for invalid in [
        'https://example.com/logo.png',
        'data:image/svg+xml;base64,PHN2Zz4=',
        'data:image/png;base64,bm90IGEgcG5n',
        'data:image/png;base64,!!!',
        'data:image/webp;base64,UklGRg==',
        'data:image/png;base64,'
        + base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'x' * 512 * 1024).decode(),
    ]:
        with pytest.raises(ValidationError):
            settings(**{field: invalid})
    assert getattr(settings(**{field: ''}), field) is None


@pytest.mark.parametrize(
    'url',
    [
        'javascript:alert(1)',
        '//evil.test',
        '/\\evil.test',
        'https://user:password@example.com',
        'https://example.com/a\nb',
        'ftp://example.com',
    ],
)
def test_unsafe_footer_links_are_rejected(url):
    with pytest.raises(ValidationError):
        settings(footer_url=url)


@pytest.mark.parametrize(
    'url', ['https://example.com/about', '/about', 'http://localhost:8080']
)
def test_footer_links(url):
    assert settings(footer_url=url).footer_url == url


@pytest.mark.parametrize(
    'overrides',
    [
        {'site_name': '  '},
        {'site_name': 'a' * 129},
        {'primary_color': 'red'},
        {'primary_color': '#fff'},
        {'id': 2},
    ],
)
def test_invalid_settings_are_rejected(overrides):
    with pytest.raises(ValidationError):
        settings(**overrides)


def test_public_response_excludes_internal_fields():
    public = SiteSettingsPublic.model_validate(SiteSettings(id=1))
    assert 'id' not in public.model_dump()
    assert public.site_name == 'GPUStack'


@pytest.mark.parametrize('dialect', [mysql.dialect(), postgresql.dialect()])
def test_assets_fit_supported_database_column_types(dialect):
    ddl = str(CreateTable(SiteSettings.__table__).compile(dialect=dialect))
    assert 'id = 1' in ddl
    assert 'MEDIUMTEXT' in ddl if dialect.name == 'mysql' else 'TEXT' in ddl


@pytest.mark.parametrize(
    'mime,signature',
    [
        ('image/png', b'\x89PNG\r\n\x1a\n'),
        ('image/jpeg', b'\xff\xd8\xff'),
        ('image/webp', b'RIFF\x00\x00\x00\x00WEBP'),
    ],
)
@pytest.mark.parametrize('size', [512 * 1024 - 1, 512 * 1024, 512 * 1024 + 1])
def test_asset_decoded_size_boundary(mime, signature, size):
    content = signature + b'\x00' * (size - len(signature))
    asset = f'data:{mime};base64,' + base64.b64encode(content).decode()
    if size > 512 * 1024:
        with pytest.raises(ValidationError, match='must not exceed 512 KiB'):
            settings(logo=asset)
    else:
        assert settings(logo=asset).logo == asset


@pytest.mark.parametrize('value', [None, ''])
def test_optional_assets_and_footer_link_can_be_cleared(value):
    result = settings(
        logo=value,
        logo_dark=value,
        mini_logo=value,
        mini_logo_dark=value,
        favicon=value,
        footer_url=value,
    )
    for field in [
        'logo',
        'logo_dark',
        'mini_logo',
        'mini_logo_dark',
        'favicon',
        'footer_url',
    ]:
        assert getattr(result, field) is None


@pytest.mark.parametrize(
    'url',
    [
        'https://',
        'https:///missing-host',
        '/a\x00b',
        '/a\x7fb',
        '/a b',
        'data:text/html,hello',
    ],
)
def test_links_require_a_host_and_reject_browser_ambiguities(url):
    with pytest.raises(ValidationError):
        settings(footer_url=url)


@pytest.mark.parametrize(
    'field,limit', [('site_name', 128), ('footer_name', 128), ('footer_url', 2048)]
)
def test_text_field_length_limits(field, limit):
    prefix = '/' if field == 'footer_url' else ''
    value = prefix + 'x' * (limit - len(prefix))
    assert getattr(settings(**{field: value}), field) == value
    with pytest.raises(ValidationError):
        settings(**{field: value + 'x'})


def test_text_is_normalized_before_saving():
    result = settings(
        site_name='  Research Lab  ',
        footer_name=' Lab Inc. ',
        footer_url=' https://example.com/about ',
        primary_color=' #aBc123 ',
    )
    assert result.site_name == 'Research Lab'
    assert result.footer_name == 'Lab Inc.'
    assert result.footer_url == 'https://example.com/about'
    assert result.primary_color == '#aBc123'


@pytest.mark.parametrize('value', [None, '', ' ', '\t\n'])
def test_blank_footer_name_clears_the_setting(value):
    assert settings(footer_name=value).footer_name is None


@pytest.mark.parametrize(
    'url',
    ['https://example.com:abc', 'https://example.com:65536', 'http://[::1]:-1'],
)
def test_invalid_footer_ports_are_rejected(url):
    with pytest.raises(ValidationError):
        settings(footer_url=url)


@pytest.mark.parametrize('url', ['http://[::1]:8080', 'https://example.com:65535'])
def test_valid_footer_ports_are_preserved(url):
    assert settings(footer_url=url).footer_url == url
