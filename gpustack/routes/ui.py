import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from gpustack.config.config import Config


def register(app: FastAPI):
    ui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
    if not os.path.isdir(ui_dir):
        raise RuntimeError(f"directory '{ui_dir}' does not exist")

    for name in ["css", "js", "static"]:
        app.mount(
            f"/{name}",
            StaticFiles(directory=os.path.join(ui_dir, name)),
            name=name,
        )

    @app.get("/", include_in_schema=False)
    async def index(request: Request):
        """
        Serve the main index.html with optional CAS login button injection.
        """
        config: Config = request.app.state.server_config
        index_path = os.path.join(ui_dir, "index.html")

        # If CAS is enabled, inject custom CSS and JS for CAS login button
        if config.cas_server_url:
            with open(index_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Inject custom CSS for CAS button styling
            cas_css = """
<style>
.gpustack-cas-login-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    padding: 12px 24px;
    margin-top: 16px;
    margin-bottom: 16px;
    background-color: #1677ff;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
}
.gpustack-cas-login-btn:hover {
    background-color: #4096ff;
    box-shadow: 0 2px 8px rgba(22, 119, 255, 0.4);
}
.gpustack-cas-login-btn .cas-icon {
    margin-right: 8px;
    font-size: 16px;
}
.gpustack-cas-divider {
    display: flex;
    align-items: center;
    margin: 16px 0;
    color: #8c8c8c;
    font-size: 12px;
}
.gpustack-cas-divider::before,
.gpustack-cas-divider::after {
    content: '';
    flex: 1;
    border-top: 1px solid #d9d9d9;
}
.gpustack-cas-divider span {
    padding: 0 12px;
}
</style>
"""
            # Inject CSS before </head>
            content = content.replace("</head>", cas_css + "</head>")

            # Inject JS to add CAS login button to the login page
            cas_js = """
<script>
(function() {
    'use strict';

    // Check if current page is the login page
    function isLoginPage() {
        const hash = window.location.hash;
        // Only show on login page or root
        return hash === '#/login' || hash === '' || hash === '#';
    }

    // Remove CAS button if not on login page
    function removeCasButton() {
        const casBtn = document.querySelector('.gpustack-cas-login-btn');
        const divider = document.querySelector('.gpustack-cas-divider');
        if (casBtn) casBtn.remove();
        if (divider) divider.remove();
    }

    // Add or remove CAS button based on current page
    function updateCasButton() {
        if (!isLoginPage()) {
            removeCasButton();
            return;
        }

        // Find the login form container
        const loginForm = document.querySelector('form') || document.querySelector('[class*="login"]') || document.getElementById('root');
        if (!loginForm) {
            setTimeout(updateCasButton, 100);
            return;
        }

        // Check if CAS button already exists
        if (document.querySelector('.gpustack-cas-login-btn')) {
            return;
        }

        // Create divider
        const divider = document.createElement('div');
        divider.className = 'gpustack-cas-divider';
        divider.innerHTML = '<span>SSO Login</span>';

        // Create CAS login button
        const casBtn = document.createElement('a');
        casBtn.href = '/auth/cas/login';
        casBtn.className = 'gpustack-cas-login-btn';
        casBtn.innerHTML = '<span class="cas-icon">🔐</span> CAS Single Sign-On';
        casBtn.onclick = function(e) {
            e.preventDefault();
            window.location.href = '/auth/cas/login';
        };

        // Insert button after the login form or at appropriate position
        const submitBtn = loginForm.querySelector('button[type="submit"]') || loginForm.querySelector('button');
        if (submitBtn && submitBtn.parentNode) {
            submitBtn.parentNode.insertBefore(divider, submitBtn.nextSibling);
            divider.parentNode.insertBefore(casBtn, divider.nextSibling);
        } else if (loginForm) {
            loginForm.appendChild(divider);
            loginForm.appendChild(casBtn);
        }
    }

    // Initial check
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', updateCasButton);
    } else {
        updateCasButton();
    }

    // Also try after a delay for SPA frameworks
    setTimeout(updateCasButton, 500);
    setTimeout(updateCasButton, 1000);
    setTimeout(updateCasButton, 2000);

    // Listen for hash changes (SPA navigation) and remove button when leaving login page
    window.addEventListener('hashchange', function() {
        if (!isLoginPage()) {
            removeCasButton();
        } else {
            setTimeout(updateCasButton, 100);
        }
    });

    // Use MutationObserver to detect SPA navigation and remove button
    const observer = new MutationObserver(function() {
        if (!isLoginPage()) {
            removeCasButton();
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""
            # Inject JS before </body>
            content = content.replace("</body>", cas_js + "</body>")

            return HTMLResponse(content=content)

        return FileResponse(index_path)