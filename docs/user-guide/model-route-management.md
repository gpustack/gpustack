# Model Route Management

GPUStack provides model route management capabilities. Through model routes, you can implement model aliases, traffic distribution, disaster recovery, and unified entry for both public and private models.

## Create Route

1. Go to `Routes` page.
2. Click `Add Route`.
3. Fill the route `Name` as the serving model name.
4. Select at lease one `Route Targets`.
5. Click the `Save` button.

## Manage Route Targets

1. Go to `Routes` page.
2. Unfold the model route you want to manage targets for.
3. Click the `Delete` button in `Operations` column for the target you don't want to keep.
4. Click the `Edit` button in the `Operations` column. On the edit route page, add or remove targets for the model route, or adjust the traffic weight for the targets.
5. On the edit route page, select or clear the fallback route target for this route.

## Authorize Route Access

1. Go to `Routes` page.
2. Find the route for which you want to change the authorization setting.
3. Click the `Access Settings` button in the `Operations` column.
4. Change the `Access Scope` as needed.
5. For the `Allowed Users` scope, select the users you want to authorize for this route and click `>` to confirm.
6. Click the `Save` button.

## Edit Route

1. Go to `Routes` page.
2. Find the route you want to edit.
3. Click the `Edit` button in the `Operations` column.
4. Modify name, model category, description and route targets as needed.
5. Click the `Save` button.

## Delete Route

1. Go to `Routes` page.
2. Find the route you want to delete.
3. Click the `Delete` button in the `Operations` column.
4. Confirm the deletion.
