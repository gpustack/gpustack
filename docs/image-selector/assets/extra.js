document.addEventListener("DOMContentLoaded", function () {
    // Find all horizontal navigation links pointing to the image selector
    let selectorLinks = document.querySelectorAll('a[href*="image-selector"]');

    const externalLinkIcon = `
        <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: text-bottom; margin-left: 4px; fill: none !important;">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
            <polyline points="15 3 21 3 21 9"></polyline>
            <line x1="10" y1="14" x2="21" y2="3"></line>
        </svg>
    `;

    selectorLinks.forEach(function (link) {
        // 1. Set to open in a new tab
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');

        // 2. Check if the icon has already been added (prevent duplicate icons if script runs multiple times)
        if (!link.innerHTML.includes('<svg')) {
            link.insertAdjacentHTML('beforeend', externalLinkIcon);
        }
    });
});