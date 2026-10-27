# Cloud Credential Management

GPUStack supports cloud credential management, allowing secure connections to external cloud providers. Cloud credentials contain provider information, keys, and options required for API access.

## Supported Providers

`DigitalOcean` and `SHUIHUA FUTURE` are supported.

## Create Cloud Credential

1. Go to the `Cloud Credentials` page.
2. Click the `Add Cloud Credential` dropdown and select a provider.
3. Fill in the following information:

   - `Name`: Unique credential name.
   - The provider's secret, which is the only credential field either one needs:
     - `DigitalOcean` — `Access Token`, the API token generated on the DigitalOcean `Applications & API` page. The token scope must be `Full Access`.
     - `SHUIHUA FUTURE` — `API Key`, created on the [Shuihua console](https://hub.do.top/cn/signin) and prefixed with `amp_live_`.
   - `Description`: Additional information for the cloud credential.

4. Click the `Save` button.

## Update Cloud Credential

1. Go to the `Cloud Credentials` page.
2. Find the credential you want to edit.
3. Click the `Edit` button.
4. Update the `Name`, `Access Token`, and `Description` as needed.
5. Click the `Save` button.

## Delete Cloud Credential

1. Go to the `Cloud Credentials` page.
2. Find the credential you want to delete.
3. Click the ellipsis button in the operations column, then select `Delete`.
4. Confirm the deletion.
