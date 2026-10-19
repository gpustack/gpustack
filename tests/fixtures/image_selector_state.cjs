const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('docs/image-selector/assets/app.js', 'utf8');

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
    }
    set innerHTML(html) {
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
            return [];
        },
        createElement(tag) { return new Element(tag); }
    };
    const context = {
        document, URL, URLSearchParams, window: {location, history},
        localStorage: {getItem() { return 'zh'; }, setItem() {}},
        console, setTimeout() {},
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
    return {...context.API, nodes, links, location, history};
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

(async () => {
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
})().catch(error => { console.error(error); process.exitCode = 1; });
