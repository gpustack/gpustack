# Site Settings

Platform administrators can customize the branding of a GPUStack deployment under **Access Control → Site settings**. Settings are shared by all users and organizations and persist in the server database across restarts and replicas.

## Customize the site

1. Sign in as a platform administrator and open **Site settings**.
2. Set the site name and primary color.
3. Optionally upload a logo, compact sidebar logo, dark-theme variants, and favicon.
4. Optionally set the footer company name and link.
5. Click **Save changes**. The current browser applies the saved settings immediately. Other browsers use them on their next reload.

The site name is used for the browser title and logo alternative text. On the login page, a custom name is displayed when no custom logo is provided. Dark-theme logos fall back to the corresponding light-theme asset. Clearing an asset restores the built-in asset; **Reset to default** restores the primary color. Empty footer fields use GPUStack's defaults.

Images must be PNG, JPEG, or WebP, at most 512 KiB each. Assets are stored in the database; external image URLs and SVG uploads are not supported. Footer links accept HTTP(S) URLs or same-origin paths such as `/about`.

Branding is public, including before sign-in. Do not include confidential information in these fields.

## API

| Method | Endpoint | Access |
| --- | --- | --- |
| GET | `/v2/site-settings/public` | Public branding fields only; no authentication required |
| GET | `/v2/site-settings` | Platform administrator |
| PUT | `/v2/site-settings` | Platform administrator with management scope |

`PUT` replaces the configuration. `site_name` (1–128 characters) and `primary_color` (a six-digit hexadecimal color, such as `#007BFF`) are required. Optional fields are `logo`, `logo_dark`, `mini_logo`, `mini_logo_dark`, `favicon`, `footer_name` (up to 128 characters), and `footer_url` (up to 2048 characters). Omitted optional fields are cleared. Image values are base64 data URLs. Responses include `updated_at`, which is null until the first change. Unknown input fields are rejected.

## Deployment compatibility

Deploy the corresponding server and UI versions together to enable editing. The server's Alembic upgrade creates the `site_settings` table in the v2.3.1 migration bundle. Before the first save, reads return GPUStack defaults without creating a row. Downgrading removes the table and its saved branding.

A UI running against an older server, or unable to load public settings within five seconds, starts with GPUStack defaults. An unsuccessful administrator settings read disables editing and offers a retry.

Builds with a branding plugin keep that plugin's branding initialization and editor. The built-in site settings page is available in builds without a branding plugin.
