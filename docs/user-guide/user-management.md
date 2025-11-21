# User Management

GPUStack has two user roles: `Admin` and `User`.

Admins can manage clusters, resources, models, users, and system settings.

Users can manage their own API keys and access the model APIs.

## Default Admin

On bootstrap, GPUStack creates a default admin user.

The initial password is saved in `<data-dir>/initial_admin_password`.

In the default setup, this file is located at `/var/lib/gpustack/initial_admin_password` inside the server container.

You can set a custom password for the default admin by using the `--bootstrap-password` flag when starting `GPUStack`.

## Create User

1. Navigate to the `Users` page.
2. Click the `Add User` button.
3. Fill in `Name`, `Full Name`, `Password`, and select `Role` for the user.
4. Click the `Save` button.

## Update User

1. Navigate to the `Users` page.
2. Find the user you want to edit.
3. Click the `Edit` button in the `Operations` column.
4. Update the attributes as needed.
5. Click the `Save` button.

## Deactivate User

1. Navigate to the `Users` page.
2. Find the user you want to deactivate.
3. Click the `Deactivate Account` button in the `Operations` column.

## Activate User

1. Navigate to the `Users` page.
2. Find the user you want to activate.
3. Click the `Activate Account` button in the `Operations` column.

## Delete User

1. Navigate to the `Users` page.
2. Find the user you want to delete.
3. Click the ellipsis button in the `Operations` column, then select `Delete`.
4. Confirm the deletion.

!!! note

    The admin user cannot be deactivated or deleted from the UI.
