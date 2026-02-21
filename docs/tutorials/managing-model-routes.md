# Managing Model Routes

This guide introduces how to use model routes, covering several common use cases and their configuration methods.

## Create route when deploying model

When deploying models, the `Enable Model Route` checkbox is automatically enabled. It will create a model route for this deployment with the same name. This allows users to access the model directly using the same name after deployment.

## Model Upgrade with Model Alias

When a new version of a model is released, the administrator may want to upgrade the model while keeping the same model name. In this case, the administrator can deploy a new version of the model with the model route disabled, and switch traffic by editing the existing model route.

1. Add enough GPU resources for the new version of the model into GPUStack.
2. Deploy the model with the new version, uncheck the `Enable Model Route` option in the model deploy drawer.
3. Locate the model route that targets the old version of the model on the `Routes` page.
4. Edit this route and replace the target with the new version model deployment.
5. New requests to this model route will be routed to the new version model.

## Serve Model for Self-host Models and Public MaaS Models

When the request volume for a self-hosted model increases, latency may occur. If there are no resources available for scaling up, introducing Public MaaS is an effective solution. By configuring both the deployment model target and the provider’s model target in the model route and assigning weights, you can use Public MaaS services to help handle the current model’s access load.

1. Go to the `Providers` page.
2. Add a provider as needed and select the models to use in the provider.
3. Go to the `Routes` page and locate the model to edit.
4. Edit the model and add route targets for this model route.
5. Both models from GPUStack `Deployments` and models from `Providers` can be selected as targets in the same route.
6. The weight (default is 100) for each target determines the traffic percentage for this route. For example, if target A has a weight of 100 and target B has a weight of 200, 33% of requests will be routed to A and 67% to B.

## Model Route Fallback

Although assigning a Public MaaS model target to a model route is a convenient approach, it can also incur significant costs. The traffic distribution rules are always in effect, so even when the self-hosted model is not under heavy load, traffic will still be forwarded to Public MaaS according to the configuration. In such cases, using the Model Route `Fallback` feature can be very effective.

1. Go to the `Routes` page and locate the model route you want to set a fallback for.
2. Edit this route and configure the `Fallback Route Target`. Like other route targets, it can be a model from GPUStack `Deployments` or from `Providers`.
3. For the fallback target, it is mutually exclusive with the traffic distribution strategy, so you cannot configure weights for this target by design.

## Proxy OpenAI Compatible Inference Service via Model Route

If a running inference service (such as ollama or lm-studio) wants to use GPUStack for proxying, access control, and token usage statistics, you can create a custom-path `OpenAI` Model Provider for hosting.

1. Go to the `Providers` page and click the `Add Provider` button.
2. Select `OpenAI` as the type and set the `Custom Base URL` in the form of `http://<ip>:<port>/v1` for your OpenAI-compatible inference server. Set the name, API key, and description as needed.
3. Add models for this provider. The available models will be listed for selection if the inference server supports the `/v1/models` API.
4. Click the `Save` button.
5. Click `Add Route` in the `Operations` column for this provider.
6. The first model of the provider will be used to pre-configure the route. Adjust the route configuration as needed.
7. Click the `Save` button to apply the model route. Your model is now proxied by GPUStack.
8. Authorize access to this route using `Access Setting` in the `Operations` column.
