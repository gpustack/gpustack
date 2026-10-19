const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('docs/image-selector/assets/app.js', 'utf8');
const html = fs.readFileSync('docs/image-selector/index.html', 'utf8');

class Element {
    constructor(tagName = 'div', dataset = {}) {
        this.tagName = tagName;
        this.dataset = dataset;
        this.style = {};
        this.children = [];
        this.events = {};
        this.classList = {add() {}, toggle() {}};
        this.textContent = '';
        this.checked = false;
        this.disabled = false;
        this.htmlWrites = [];
    }
    set innerHTML(html) {
        this.htmlWrites.push(html);
        this.children = [];
        for (const match of html.matchAll(/<(option|input)\b([^>]*)>/g)) {
            const element = new Element(match[1]);
            element.value = /value="([^"]*)"/.exec(match[2])?.[1] || '';
            this.appendChild(element);
        }
    }
    set selected(value) {
        this._selected = value;
        if (value && this.parent?.tagName === 'select') {
            this.parent.children.forEach(child => {
                if (child !== this) child._selected = false;
            });
        }
    }
    get selected() { return this._selected; }
    get selectedOptions() { return this.children.filter(child => child.selected); }
    appendChild(child) {
        child.parent = this;
        this.children.push(child);
        if (this.tagName === 'select' && !this.selectedOptions.length) child.selected = true;
        return child;
    }
    querySelectorAll(selector) {
        const descendants = this.children.flatMap(child => [child, ...child.querySelectorAll('*')]);
        return descendants.filter(child => {
            if (selector === '*') return true;
            if (selector === '.option-button') return child.className === 'option-button';
            if (selector === 'option') return child.tagName === 'option';
            if (selector === 'option:not([value=""])') return child.tagName === 'option' && child.value;
            if (selector === 'input') return child.tagName === 'input';
            if (selector === 'input:checked') return child.tagName === 'input' && child.checked;
            return child.dataset.value === /\[data-value="([^"]*)"\]/.exec(selector)?.[1];
        });
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    addEventListener(name, callback) { this.events[name] = callback; }
}

function app(language, href, options = {}) {
    const nodes = {};
    const created = [];
    const translations = Array.from(html.matchAll(/<[^>]+data-i18n="([^"]+)"[^>]*>/g),
        match => new Element('div', {i18n: match[1]}));
    const links = ['en', 'zh', 'ja', 'ru', 'tr'].map(lang => {
        const link = new Element('a', {lang});
        link.href = `https://docs.example/help/${lang === 'en' ? '' : `${lang}/`}image-selector/`;
        return link;
    });
    const location = new URL(href);
    location.assign = href => { location.assigned = href; };
    const history = {
        replaced: [],
        replaceState(state, title, href) {
            this.replaced.push(href);
            location.href = href;
        }
    };
    const document = {
        documentElement: {dataset: {docsLanguage: language}, lang: language},
        body: new Element(),
        addEventListener() {},
        getElementById(id) {
            if (id === 'copy-images-command') return null;
            return nodes[id] ||= new Element(id.endsWith('-select') ? 'select' : 'div');
        },
        querySelector() { return null; },
        querySelectorAll(selector) {
            if (selector === '.dropdown-content a') return links;
            if (selector === '[data-i18n]') return translations;
            return [];
        },
        createElement(tag) {
            const element = new Element(tag);
            created.push(element);
            return element;
        }
    };
    const context = {
        document, URL, URLSearchParams, window: {location, history},
        localStorage: {getItem() { return 'zh'; }, setItem() {}},
        console: options.console || console, setTimeout() {},
        fetch: options.fetch || (async url => ({
            json: async () => JSON.parse(fs.readFileSync(`docs/image-selector/${url.replace(/^\.\//, '')}`, 'utf8'))
        }))
    };
    vm.createContext(context);
    vm.runInContext(source + `
        globalThis.API = {
            init, state, selectGpuStackVersion, selectCard, selectFrameworkVersion,
            selectArch, selectRegistry, selectComponent, updateSelectedBackends,
            switchGuideTab, setLanguage
        };
    `, context);
    const htmlWrites = () => [...Object.values(nodes), ...created, ...translations]
        .flatMap(element => element.htmlWrites);
    return {...context.API, nodes, links, location, history, htmlWrites, translations};
}

const selection = state => JSON.parse(JSON.stringify({
    selectedGpuStackVersion: state.selectedGpuStackVersion,
    selectedComponent: state.selectedComponent,
    selectedCard: state.selectedCard,
    selectedFrameworkVersion: state.selectedFrameworkVersion,
    selectedChipType: state.selectedChipType,
    selectedArch: state.selectedArch,
    selectedRegistry: state.selectedRegistry,
    selectedBackends: state.selectedBackends,
    optionalImages: state.optionalImages,
    selectedGuide: state.selectedGuide
}));
const commands = page => page.nodes['image-list'].textContent.split('\n').filter(line => line.startsWith('docker '));

async function checkStateRestoration() {
    const saved = {
        selectedGpuStackVersion: 'v2.2.2', selectedComponent: 'worker',
        selectedCard: 'ascend', selectedFrameworkVersion: '8.2', selectedChipType: '910b',
        selectedArch: 'arm64', selectedRegistry: 'quay',
        selectedBackends: ['vllm-0.10.2'],
        optionalImages: {postgres: true, monitoring: true}, selectedGuide: 'tag-push'
    };
    const href = new URL('https://docs.example/help/ja/image-selector/?keep=one#offline');
    href.searchParams.set('selector', JSON.stringify(saved));
    const restored = app('ja', href.href);
    await restored.init();
    assert.deepEqual(selection(restored.state), saved, 'URL configuration survives asynchronous manifest loading');
    assert.equal(restored.state.currentLang, 'ja', 'target page controls language');
    assert.ok(commands(restored).every(line => line.includes('--platform linux/arm64')));
    assert.ok(commands(restored).some(line => line.includes('cann8.2-910b-vllm0.10.2')));
    assert.ok(!commands(restored).some(line => line.includes('cuda')));
    assert.ok(restored.nodes.postgres.checked && restored.nodes.monitoring.checked);

    const page = app('en', 'https://docs.example/help/image-selector/?keep=two#download');
    await page.init();
    await page.selectGpuStackVersion('v2.2.2');
    page.selectCard('ascend');
    const framework = page.nodes['framework-version-select'].children.find(option =>
        option.value === '8.2' && option.dataset.chipType === '910b');
    framework.selected = true;
    page.selectFrameworkVersion(framework.value);
    const backend = page.nodes['backend-selector'].querySelectorAll('input').find(input => input.value === 'vllm-0.10.2');
    backend.checked = true;
    page.updateSelectedBackends();
    page.selectArch('arm64');
    page.selectRegistry('quay');
    page.selectComponent('server');
    for (const key of ['postgres', 'monitoring']) {
        page.nodes[key].checked = true;
        page.nodes[key].events.change();
    }
    page.switchGuideTab('copy-images');
    const before = selection(page.state);
    const beforeCommands = commands(page);
    assert.ok(beforeCommands.some(line => line.includes('postgres:')));
    assert.ok(beforeCommands.some(line => line.includes('prometheus:')));
    const copied = app('ru', page.location.href);
    await copied.init();
    assert.deepEqual(selection(copied.state), before, 'copied URL restores every selectable field');
    assert.deepEqual(commands(copied), beforeCommands, 'copied URL generates identical image commands');

    for (const language of ['zh', 'ja', 'ru', 'tr', 'en']) {
        const link = page.links.find(link => link.dataset.lang === language);
        let prevented = false;
        link.events.click({preventDefault() { prevented = true; }});
        assert.ok(prevented);
        const destination = new URL(page.location.assigned);
        assert.equal(destination.searchParams.get('keep'), 'two');
        assert.equal(destination.hash, '#download');
        const translated = app(language, destination.href);
        await translated.init();
        assert.deepEqual(selection(translated.state), before, `${language} preserves all choices`);
        assert.deepEqual(commands(translated), beforeCommands, `${language} preserves commands`);
        assert.equal(translated.state.currentLang, language);
    }

    const link = page.links[0];
    for (const href of ['https://external.example/help/image-selector/', 'https://docs.example:8443/help/image-selector/', 'https://docs.example/settings/']) {
        link.href = href;
        page.location.assigned = null;
        let prevented = false;
        link.events.click({preventDefault() { prevented = true; }});
        assert.equal(prevented, false, 'external language links use ordinary navigation');
        assert.equal(page.location.assigned, null);
    }

    const chineseDefault = app('zh', 'https://docs.example/help/zh/image-selector/');
    await chineseDefault.init();
    assert.equal(chineseDefault.state.selectedRegistry, 'china', 'Chinese default applies without a saved choice');

    for (const registry of ['docker-hub', 'quay', 'china']) {
        const explicit = app('zh', href.href);
        await explicit.init();
        explicit.selectRegistry(registry);
        explicit.setLanguage('en');
        assert.equal(explicit.state.selectedRegistry, registry, 'language never overrides a configured registry');
        assert.equal(explicit.state.selectedArch, 'arm64');
        assert.equal(explicit.state.selectedFrameworkVersion, '8.2');
        assert.deepEqual(Array.from(explicit.state.selectedBackends), ['vllm-0.10.2']);
    }

    for (const raw of ['invalid JSON', 'null', '[]', JSON.stringify({
        selectedGpuStackVersion: 'removed', selectedComponent: 'invalid', selectedCard: 'missing',
        selectedArch: 'invalid', selectedRegistry: 'missing', optionalImages: null
    })]) {
        const bad = app('en', `https://docs.example/help/image-selector/?selector=${encodeURIComponent(raw)}`);
        await bad.init();
        assert.equal(bad.state.selectedGpuStackVersion, 'v2.2.3');
        assert.equal(bad.state.selectedCard, 'nvidia');
        assert.equal(bad.state.selectedArch, 'amd64');
        assert.equal(bad.state.selectedRegistry, 'docker-hub');
        assert.equal(bad.state.selectedComponent, 'all');
        assert.ok(commands(bad).length);
    }
    const obsolete = {...saved, selectedFrameworkVersion: 'removed', selectedChipType: 'removed', selectedBackends: ['missing']};
    const fallback = app('en', `https://docs.example/help/image-selector/?selector=${encodeURIComponent(JSON.stringify(obsolete))}`);
    await fallback.init();
    assert.notEqual(fallback.state.selectedFrameworkVersion, 'removed');
    assert.deepEqual(Array.from(fallback.state.selectedBackends), []);
    assert.ok(commands(fallback).some(line => line.includes('runner:cann')));
    assert.ok(page.history.replaced.length, 'configuration updates replace the current history entry');
}

async function checkSelectorStateSafety() {
    const marker = '<span data-selector-marker="url"></span>';
    const poisoned = {
        selectedGpuStackVersion: marker,
        selectedComponent: marker, selectedCard: marker,
        selectedFrameworkVersion: marker, selectedChipType: marker,
        selectedArch: marker, selectedRegistry: marker,
        selectedBackends: [marker], optionalImages: {postgres: marker, monitoring: marker},
        selectedGuide: marker, images: [marker], unknown: marker
    };
    const url = language => `https://docs.example/help/${language}/image-selector/?selector=${encodeURIComponent(JSON.stringify(poisoned))}`;
    const safeHTML = page => {
        assert.ok(page.htmlWrites().length, 'real translated guides are rendered');
        assert.ok(page.htmlWrites().every(value => !value.includes(marker)),
            'untrusted markup never reaches any innerHTML assignment');
    };

    for (const language of ['en', 'zh', 'ja', 'ru', 'tr']) {
        for (const phase of ['versions', 'images']) {
            for (const fails of [false, true]) {
                let resolveManifest;
                let rejectManifest;
                let requested;
                const waiting = new Promise(resolve => { requested = resolve; });
                const pendingManifest = new Promise((resolve, reject) => {
                    resolveManifest = resolve;
                    rejectManifest = reject;
                });
                const requests = [];
                const page = app(language, url(language), {
                    console: {error() {}},
                    fetch: async path => {
                        requests.push(path);
                        if ((path === './versions/index.json') === (phase === 'versions')) {
                            requested();
                            return pendingManifest;
                        }
                        return {json: async () => JSON.parse(fs.readFileSync(`docs/image-selector/${path.replace(/^\.\//, '')}`, 'utf8'))};
                    }
                });
                const initialized = page.init();
                await waiting;
                safeHTML(page);
                assert.equal(page.state.selectedGpuStackVersion, phase === 'versions' ? null : 'v2.2.3',
                    'URL version is not live before actual version-list validation');
                page.setLanguage(language === 'ja' ? 'ru' : 'ja');
                safeHTML(page);
                if (fails) rejectManifest(Error('manifest unavailable'));
                else resolveManifest({json: async () => phase === 'versions'
                    ? {versions: ['v2.2.3', 'v2.2.2']}
                    : JSON.parse(fs.readFileSync('docs/image-selector/versions/v2.2.3.json', 'utf8'))});
                await initialized;
                safeHTML(page);
                assert.ok(requests.every(path => !path.includes(marker)));
                assert.equal(page.state.selectedGpuStackVersion, fails && phase === 'versions' ? null : 'v2.2.3');
                assert.ok(!JSON.stringify(selection(page.state)).includes(marker),
                    'failed or completed initialization cannot retain unvalidated URL fields');
            }
        }
    }

    for (const manifest of [null, {}, [], {versions: marker}, {versions: [null]}, {versions: []}]) {
        const page = app('en', url('en'), {
            console: {error() {}},
            fetch: async () => ({json: async () => manifest})
        });
        await page.init();
        safeHTML(page);
        assert.equal(page.state.selectedGpuStackVersion, null, 'malformed version lists cannot validate URL candidates');
        assert.deepEqual(Array.from(page.state.images), []);
    }

    const badImages = app('en', url('en'), {
        console: {error() {}},
        fetch: async path => ({json: async () => path.endsWith('index.json')
            ? {versions: ['v2.2.3']} : [null, {image: marker}]})
    });
    await badImages.init();
    safeHTML(badImages);
    assert.deepEqual(Array.from(badImages.state.images), [], 'malformed images cannot become live state');

    const tokens = "$&|$$|$`|$'|{{registry}}";
    const manifestMarker = '<span data-selector-marker="manifest"></span>';
    const images = [
        'gpustack/runner:cuda13.0-vllm0.25.1',
        `gpustack/runtime:pause-${manifestMarker}-${tokens}`
    ];
    const version = `v2.2.3-${manifestMarker}-${tokens}`;
    for (const language of ['en', 'zh', 'ja', 'ru', 'tr']) {
        const page = app(language, 'https://docs.example/help/image-selector/', {
            fetch: async path => ({json: async () => path.endsWith('index.json') ? {versions: [version]} : images})
        });
        await page.init();
        const rendered = page.htmlWrites().join('\n');
        assert.ok(!rendered.includes(manifestMarker), 'dynamic manifest text is HTML encoded');
        assert.ok(rendered.includes('&lt;span data-selector-marker=&quot;manifest&quot;&gt;&lt;/span&gt;'));
        assert.ok(rendered.includes('$&amp;|$$|$`|$&#39;|{{registry}}'),
            'replacement tokens and nested-looking placeholders remain literal text');
        assert.ok(commands(page).some(command => command.includes(manifestMarker)),
            'image commands remain plain text rather than executable markup');
        assert.ok(!rendered.includes('&amp;lt;Server'), 'static placeholder entities are not escaped twice');
    }
}

module.exports = {checkStateRestoration, checkSelectorStateSafety};
if (require.main === module) {
    Promise.resolve().then(checkStateRestoration).then(checkSelectorStateSafety)
        .catch(error => { console.error(error); process.exitCode = 1; });
}
