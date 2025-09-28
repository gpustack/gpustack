# User Management

GPUStack supports users of two roles: `Admin` and `User`. Admins can monitor system status, manage models, users, and system settings. Users can manage their own API keys and use the model API.

## Default Admin

On bootstrap, GPUStack creates a default admin user. The initial password for the default admin is stored in `<data-dir>/initial_admin_password`. In the default setup, it should be `/var/lib/gpustack/initial_admin_password`. You can customize the default admin password by setting the `--bootstrap-password` parameter when starting `gpustack`.

## Create User

1. Navigate to the `Users` page.
2. Click the `Create User` button.
3. Fill in `Name`, `Full Name`, `Password`, and select `Role` for the user.
4. Click the `Save` button.

## Update User

1. Navigate to the `Users` page.
2. Find the user you want to edit.
3. Click the `Edit` button in the `Operations` column.
4. Update the attributes as needed.
5. Click the `Save` button.

## Delete User

1. Navigate to the `Users` page.
2. Find the user you want to delete.
3. Click the ellipsis button in the `Operations` column, then select `Delete`.
4. Confirm the deletion.
