## SSO Authentication
1. Add config external auth info to environment variable 
   Examples such as OIDC certification
``` 
"GPUSTACK_EXTERANL_AUTH_TYPE": "OIDC"
"GPUSTACK_EXTERANL_AUTH_NAME": ""  # Mapping of OIDC user information to local username 
                                      for example "preferred_username"
"GPUSTACK_EXTERANL_AUTH_FULLNAME": "" # Mapping of OIDC user information to local full_name 
                                         Multiple elements combined for use + splicing for 
                                         example "name" or "firstName+lastName"
"GPUSTACK_OIDC_CLIENT_ID": "your CLIENT ID"
"GPUSTACK_OIDC_CLIENT_SECRET": "your CLIENT SECRET"
"GPUSTACK_OIDC_REDIRECT_URL": "{your server url}/auth/oidc/callback"
"GPUSTACK_OIDC_BASE_ENTRYPOINT": "sso server url"
```  
   Examples such as SAML certification
```
"GPUSTACK_EXTERANL_AUTH_TYPE": "SAML"
"GPUSTACK_EXTERANL_AUTH_NAME": ""     # Mapping of SAML user information to local username 
                                         for example "username"
"GPUSTACK_EXTERANL_AUTH_FULLNAME": "" # Mapping of SAML user information to local full_name 
                                         Multiple elements combined for use + splicing for example
                                         "fullName" or "firstName+lastName"
"GPUSTACK_SAML_SP_ENTITYID": "your sp_entityId" 
"GPUSTACK_SAML_SP_ASC_URL": "{your server url}/auth/saml/callback"
"GPUSTACK_SAML_IDP_ENTITYID": "SP Public Key"
"SAML_SP_PRIVATEKEY": "sp PrivateKey"
"SAML_IDP_ENTITYID": "your idp_entityId"
"SAML_IDP_SERVER_URL": "your idp_server_url"
"SAML_IDP_X509CERT": "IDP Public Key"
"SAML_SECURITY": {} # Signature configuration
```
2. Add external user configuration using CLI method 
   Examples such as OIDC certification
``` 
--exteranl_auth_type OIDC 
--exteranl_auth_name ""  # Mapping of OIDC user information to local username 
                                      for example "preferred_username"
--exteranl_auth_fullname "", # Mapping of OIDC user information to local full_name 
                                         Multiple elements combined for use + splicing for 
                                         example "name" or "firstName+lastName"
--oidc_client_id "your CLIENT ID"
--oidc_client_secret "your CLIENT SECRET"
--oidc_redirect_uri "{your server url}/auth/oidc/callback"
--oidc_base_entrypoint "sso server url"
```  
   Examples such as SAML certification
```
--exteranl_auth_type OIDC 
--exteranl_auth_name ""  # Mapping of OIDC user information to local username 
                                      for example "preferred_username"
--exteranl_auth_fullname "", # Mapping of OIDC user information to local full_name 
                                         Multiple elements combined for use + splicing for 
                                         example "name" or "firstName+lastName"
--saml_sp_entity_id "your sp_entityId",  
--saml_sp_asc_url "{your server url}/auth/saml/callback", 
--saml_sp_x509cert "SP Public Key",
--saml_sp_privateKey "sp PrivateKey",
--saml_idp_entity_id "your idp_entityId", 
--saml_idp_server_url "your idp_server_url",
--saml_idp_x509cert "IDP Public Key",
--saml_security {} # Signature configuration
```
