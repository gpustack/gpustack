# API Key Management

GPUStack supports authentication using API keys. Each GPUStack user can generate and manage their own API keys.

## Create API Key

1. Navigate to the `Access Control` > `API Keys` page.
2. Click the `Add API Key` button.
3. Fill in the `Name`, `Description`, and select the `Expiration` of the API key.
4. Select the `Type` of the API key:

   - **Auto-generated**: GPUStack generates the API key for you.
   - **Custom**: Provide your own key value in the `Key` field. This field is required when choosing **Custom**. Choose a value with the strength of a password you would never have to remember — see the note below.

5. In the `Access Permissions` section, select the permissions for the API key:

   - **Platform Management**: Grants access to platform management endpoints (users, models, workers, etc.).
   - **Model Access**: Grants access to inference APIs. When selected, choose either `All models` or `Allowed models`, and if choosing `Allowed models`, select which models this API key can access from the list.

   If no permission is selected, the API key has no access permissions.
6. Click the `Save` button.
7. Copy and store the key somewhere safe, then click the `Done` button.

!!! note

    The full API key value is shown only once upon creation; afterwards only a masked value is displayed. Custom keys are the exception — you already know their value.

!!! warning "Custom keys carry the strength you give them"

    GPUStack accepts any string as a custom key and cannot tell a random one
    from `123456`. Auto-generated keys are 128 bits from a cryptographic random
    source, and nothing weaker is ever created by accident.

    That difference matters beyond the obvious guessing: to authenticate keys at
    the API gateway rather than at the server, GPUStack publishes a fast hash of
    each key into the gateway's configuration, which is visible to anyone who can
    read it there and travels in diagnostic bundles. For a custom key that hash
    is unsalted and depends on nothing but the secret, so it is the same value in
    every deployment — a precomputed table built once works against all of them.
    A random key is unaffected, since searching for it is infeasible either way,
    while a guessable one becomes cheap to recover offline.

    Generate custom keys from a random source wherever you can. Where people
    choose them by hand, set `GPUSTACK_GATEWAY_AUTH_ALLOW_CUSTOM_KEYS=false`.
    Custom keys then authenticate at the server on every request, as they did
    before, and no hash of them is published — the gateway is told only that a
    key with that id exists and when it expires, which is what keeps revoking one
    working while the server is down.

    The setting applies to keys that already exist, not only to new ones: turning
    it off withdraws the hashes of custom keys created while it was on, at the
    next reconcile.

## Edit Access Permissions

1. Navigate to the `Access Control` > `API Keys` page.
2. Find the API key you want to edit.
3. Click the `Edit` button in the `Operations` column.
4. Update the `Description` if needed.
5. In the `Access Permissions` section, adjust the selected permissions:

   - **Platform Management**: Grants access to platform management endpoints (users, models, workers, etc.).
   - **Model Access**: Grants access to inference APIs. When selected, choose either `All models` or `Allowed models`, and if choosing `Allowed models`, select which models this API key can access from the list.

6. Click the `Save` button.

!!! note

    Changes will take effect within one minute.

## Delete API Key

1. Navigate to the `Access Control` > `API Keys` page.
2. Find the API key you want to delete.
3. Click the `Delete` button in the `Operations` column.
4. Confirm the deletion.

## Revoke a Key While the Server Is Unavailable

Deleting a key normally is enough: the server tells the API gateway, and the key
stops being accepted within seconds. This section is only for the case where a
key must stop working **now** and the GPUStack server is down — during an
upgrade, for instance.

The gateway keeps its own copy of the keys it may accept, which is what lets
inference keep serving while the server is unavailable. That copy lives in
Kubernetes, not in the server, so it can be edited without one:

```bash
kubectl -n higress-system edit wasmplugin gpustack-llm-ext-auth
```

Remove the key's entry under `spec.defaultConfig.local_auth`. Almost always it is
in `keys`, indexed by the key's access key — the middle segment of
`gpustack_<access key>_<secret>` for an auto-generated key. A custom key has no
access key inside it, so search `keys` for the numeric `user_id` of its owner
instead, or match the `exp` value against the key's expiration.

Where `GPUSTACK_GATEWAY_AUTH_ALLOW_CUSTOM_KEYS` is set to `false`, custom keys
are in `refs` instead, indexed by the key's numeric id as shown in the API.

A key absent from both is already asking the server on every request, so there is
nothing to remove — it stops working the moment the server is back and processes
the deletion.

The change reaches the gateway within seconds of being saved — it travels the
same way any other change to the resource does. Sessions already in flight are
covered too: everything the gateway accepts without asking the server is
re-checked against these entries first, so removing one invalidates it
everywhere at once.

!!! warning

    This is a stopgap, not a revocation. The server rebuilds this list from its
    database within seconds of starting up, so **a key removed only here comes
    back when the server does**. Once the server is available, delete the key
    the normal way as well.

!!! note

    Editing the resource requires access to the cluster the gateway runs in, and
    `higress-system` above is the default gateway namespace — substitute yours if
    it was changed. Modifying the GPUStack database directly is not a supported
    alternative and can leave the deployment inconsistent.

!!! warning "Who can read this namespace"

    The same resource holds the key the gateway signs its internal identity
    tokens with. Anyone who can read resources in the gateway's namespace can
    therefore mint one and be accepted as any user, so treat read access there
    as platform-level authority and grant it accordingly.

## Use API Key

GPUStack supports using the API key as a bearer token. The following is an example using curl:

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'
```
