"""Preserve commands and deep links when switching documentation languages."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_selector_languages_preserve_commands_and_navigation():
    _run_node(SELECTOR_CHECK)


def test_docs_language_links_preserve_version_page_and_search_state():
    _run_node(DOCS_LANGUAGE_CHECK)


def test_selector_restores_configuration_after_language_navigation():
    _run_selector_fixture("checkStateRestoration")


def test_selector_retries_saved_configuration_after_manifest_failure():
    _run_selector_fixture("checkManifestFailureRecovery")


def test_selector_url_configuration_cannot_inject_html():
    _run_selector_fixture("checkSelectorStateSafety")


@pytest.mark.parametrize("check", ["checkStateRestoration", "checkSelectorStateSafety"])
def test_selector_accepts_a_new_latest_release(tmp_path, check):
    selector = tmp_path / "docs/image-selector"
    shutil.copytree(REPOSITORY_ROOT / "docs/image-selector", selector)
    fixture = tmp_path / "tests/fixtures/image_selector_state.cjs"
    fixture.parent.mkdir(parents=True)
    shutil.copyfile(
        REPOSITORY_ROOT / "tests/fixtures/image_selector_state.cjs", fixture
    )

    index_path = selector / "versions/index.json"
    index = json.loads(index_path.read_text())
    latest_version = "v999.0.0"
    shutil.copyfile(
        selector / "versions" / f"{index['versions'][0]}.json",
        selector / "versions" / f"{latest_version}.json",
    )
    index["versions"].insert(0, latest_version)
    index_path.write_text(json.dumps(index))

    _run_selector_fixture(check, tmp_path)


def _run_selector_fixture(check, repository_root=REPOSITORY_ROOT):
    _run_node(
        'require("./tests/fixtures/image_selector_state.cjs").'
        + check
        + '().catch(error => { console.error(error); process.exitCode = 1; });',
        repository_root,
    )


def _run_node(script, repository_root=REPOSITORY_ROOT):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the image selector")

    result = subprocess.run(
        [node],
        input=script,
        cwd=repository_root,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


SELECTOR_CHECK = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('docs/image-selector/assets/app.js', 'utf8');
const html = fs.readFileSync('docs/image-selector/index.html', 'utf8');

class Node {
    constructor(dataset = {}) {
        this.dataset = dataset;
        this.style = {};
        this.events = {};
        this.children = [];
        this.innerHTML = '';
        this.textContent = '';
        this.classList = {add() {}, toggle() {}};
    }
    querySelectorAll() { return []; }
    querySelector() { return null; }
    addEventListener(name, fn) { this.events[name] = fn; }
    appendChild(node) { this.children.push(node); }
}

function app(docsLanguage, storedLanguage) {
    const nodes = {};
    const translations = [];
    const titles = [new Node({i18nTitle: 'copy_all'})];
    const links = [];
    const translatedTags = /<[^>]+data-i18n="([^"]+)"[^>]*>/g;
    for (const tag of html.matchAll(translatedTags)) {
        translations.push(new Node({i18n: tag[1]}));
    }
    for (const lang of ['en', 'zh', 'ja', 'ru', 'tr']) {
        const node = new Node({lang});
        node.href = `https://docs.example/${lang}/image-selector/`;
        links.push(node);
    }
    const description = {content: ''};
    const location = new URL('https://docs.example/ja/image-selector/?arch=arm64#offline');
    location.assign = function (url) { this.assigned = url; };
    const storage = {
        lang: storedLanguage,
        getItem(key) { return this[key]; },
        setItem(key, value) { this[key] = value; }
    };
    const document = {
        documentElement: {
            dataset: docsLanguage ? {docsLanguage} : {}, lang: ''
        },
        title: '', body: new Node(), addEventListener() {},
        getElementById(id) {
            if (id === 'copy-images-command') return null;
            return nodes[id] ||= new Node();
        },
        querySelector(query) {
            return query === 'meta[name="description"]' ? description : null;
        },
        querySelectorAll(query) {
            if (query === '[data-i18n]') return translations;
            if (query === '[data-i18n-title]') return titles;
            if (query === '.dropdown-content a') return links;
            return [];
        },
        createElement() { return new Node(); }
    };
    const context = {
        document, localStorage: storage, window: {location}, URL, URLSearchParams,
        console: {error() {}}, setTimeout() {},
        fetch: async () => { throw Error('offline'); }
    };
    vm.createContext(context);
    const exports = `
        globalThis.API = {
            CONFIG, state, elements, initElements, bindEvents, setLanguage,
            updateLanguage, generateImageList, renderFrameworkVersions,
            renderCardSelector, loadData
        };
    `;
    vm.runInContext(source + exports, context);
    const api = context.API;
    api.initElements();
    api.state.supportMatrix = {frameworks: {}};
    return {
        ...api, document, description, translations, titles, nodes,
        links, location, storage
    };
}

const page = app('ja', 'zh');
assert.equal(page.state.currentLang, 'ja', 'page language overrides preference');
assert.equal(app(null, 'ja').state.currentLang, 'ja', 'standalone JA preference');
assert.equal(app(null, 'unknown').state.currentLang, 'en', 'unsupported locale');
for (const language of ['ru', 'tr']) {
    assert.equal(app(language, 'en').state.currentLang, language);
    assert.equal(app(null, language).state.currentLang, language);
    assert.ok(html.includes(`data-lang="${language}"`));
}
const keys = object => Object.keys(object).sort();
for (const language of ['ja', 'zh', 'ru', 'tr']) {
    const localized = page.CONFIG.i18n[language];
    const english = page.CONFIG.i18n.en;
    assert.deepEqual(keys(localized), keys(english), `complete ${language} labels`);
    assert.deepEqual(keys(localized.comments), keys(english.comments));
    assert.deepEqual(keys(localized.cards), keys(english.cards));
}

for (const key of ['guide_save_content', 'guide_tag_content', 'guide_auto_content']) {
    const codeTags = /<code(?:\s[^>]*)?>([\s\S]*?)<\/code>/g;
    const commands = text => Array.from(text.matchAll(codeTags), item => item[1]);
    const links = text => Array.from(
        text.matchAll(/href="([^"]+)"/g), item => item[1]
    );
    const english = page.CONFIG.i18n.en[key];
    for (const language of ['ja', 'ru', 'tr']) {
        const localized = page.CONFIG.i18n[language][key];
        assert.deepEqual(commands(localized), commands(english),
            `${language} commands: ${key}`);
        assert.deepEqual(links(localized), links(english),
            `${language} guide links: ${key}`);
    }
}

page.updateLanguage();
assert.equal(page.document.documentElement.lang, 'ja');
assert.equal(page.document.title, page.CONFIG.i18n.ja.title);
assert.equal(page.nodes['current-lang'].textContent, '日本語');
assert.equal(page.description.content, page.CONFIG.i18n.ja.description);
assert.equal(page.nodes['registry-china'].style.display, 'none');
assert.equal(page.titles[0].title, 'すべてのコマンドをコピー');
assert.ok(page.translations.every(node => node.innerHTML), 'all labels render');
const saveGuide = page.translations.find(
    node => node.dataset.i18n === 'guide_save_content'
);
assert.ok(saveGuide.innerHTML.includes('&lt;Server イメージ一覧&gt;'));
page.bindEvents();
page.links[2].events.click({preventDefault() {}});
const languageTarget = new URL(page.location.assigned);
assert.equal(languageTarget.pathname, '/ja/image-selector/');
assert.equal(languageTarget.searchParams.get('arch'), 'arm64');
assert.equal(languageTarget.hash, '#offline');
assert.equal(JSON.parse(languageTarget.searchParams.get('selector')).selectedArch, 'amd64');
for (const language of ['ru', 'tr']) {
    page.links.find(link => link.dataset.lang === language).events.click({
        preventDefault() {}
    });
    const destination = new URL(page.location.assigned);
    assert.equal(destination.pathname, `/${language}/image-selector/`);
    assert.equal(destination.searchParams.get('arch'), 'arm64');
    assert.equal(destination.hash, '#offline');
}

const standalone = app(null, 'zh');
standalone.state.selectedRegistry = 'china';
standalone.setLanguage('ja');
assert.equal(standalone.state.currentLang, 'ja');
assert.equal(standalone.state.selectedRegistry, 'china');
assert.equal(standalone.nodes['registry-china'].style.display, 'block');
standalone.setLanguage('zh');
assert.equal(standalone.state.selectedRegistry, 'china');
assert.equal(standalone.nodes['registry-china'].style.display, 'block');
for (const language of ['ru', 'tr']) {
    standalone.setLanguage(language);
    const labels = standalone.CONFIG.i18n[language];
    assert.equal(standalone.document.documentElement.lang, language);
    assert.equal(standalone.nodes['current-lang'].textContent, labels.language_name);
    assert.equal(standalone.document.title, labels.title);
    assert.equal(standalone.description.content, labels.description);
    assert.equal(standalone.nodes['registry-china'].style.display, 'block');
    assert.ok(standalone.translations.every(node => node.innerHTML));
}
standalone.setLanguage('ja');
standalone.renderFrameworkVersions('cuda', 'nvidia');
assert.ok(
    standalone.nodes['framework-version-select'].innerHTML.includes(
        'バージョンを選択してください'
    )
);
standalone.renderCardSelector();
const cambricon = standalone.nodes['card-selector'].children.find(
    node => node.dataset.value === 'cambricon'
);
assert.ok(cambricon.innerHTML.includes('お問い合わせください'));
assert.equal(cambricon.title, '利用可能なイメージがありません');

Object.assign(page.state, {
    selectedGpuStackVersion: 'test', selectedCard: 'nvidia',
    selectedFrameworkVersion: '12.8', selectedBackends: [],
    images: [
        'gpustack/runtime:pause', 'gpustack/benchmark-runner:1', 'postgres:16',
        'prom/prometheus:v3', 'grafana/grafana:12',
        'gpustack/mirrored-higress:v1', 'gpustack/gpustack-operator:v1'
    ],
    runnerImages: ['gpustack/runner:cuda12.8-vllm0.9.0'],
    optionalImages: {postgres: true, monitoring: true}
});
for (const component of ['all', 'server', 'worker']) {
    for (const arch of ['amd64', 'auto']) {
        for (const registry of ['docker-hub', 'quay', 'china']) {
            Object.assign(page.state, {
                selectedComponent: component, selectedArch: arch,
                selectedRegistry: registry
            });
            const commands = [];
            for (const language of ['en', 'zh', 'ja', 'ru', 'tr']) {
                page.state.currentLang = language;
                page.generateImageList();
                commands.push(
                    page.nodes['image-list'].textContent.split('\n').filter(
                        line => line.startsWith('docker ')
                    )
                );
            }
            for (const translated of commands.slice(1)) {
                assert.deepEqual(commands[0], translated);
            }
        }
    }
}

(async () => {
    await standalone.loadData();
    assert.equal(
        standalone.document.body.children.at(-1).textContent,
        'データを読み込めませんでした'
    );
})();
"""


DOCS_LANGUAGE_CHECK = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('docs/image-selector/assets/extra.js', 'utf8');
const events = {};
const location = new URL(
    'https://docs.example/v2.3.0/ja/user-guide/usage/?q=gpu%20memory&view=all#limits'
);

class Element {
    constructor(href, languageLink = true, parent = null) {
        this.href = href;
        this.languageLink = languageLink;
        this.parent = parent;
    }
    closest(selector) {
        if (selector === '.md-select__link[hreflang]' && this.languageLink) {
            return this;
        }
        return this.parent ? this.parent.closest(selector) : null;
    }
}

const document = {
    addEventListener(name, callback) { events[name] = callback; },
    querySelectorAll() { return []; }
};
vm.runInNewContext(source, {document, window: {location}, URL, Element});
events.DOMContentLoaded();

for (const language of ['en', 'zh', 'ja', 'ru', 'tr']) {
    const languagePath = language === 'en' ? '' : `${language}/`;
    const path = `/v2.3.0/${languagePath}user-guide/usage/`;
    const link = new Element(`https://docs.example${path}?old=1#old`);
    events.click({target: link});
    assert.equal(
        link.href,
        `https://docs.example${path}${location.search}${location.hash}`,
        `${language} preserves its target version/page and current search state`
    );
}

const parentLink = new Element('https://docs.example/latest/zh/overview/');
const nestedTarget = new Element(undefined, false, parentLink);
events.click({target: nestedTarget});
assert.equal(
    parentLink.href,
    'https://docs.example/latest/zh/overview/' + location.search + location.hash,
    'clicking nested content still follows the language link'
);

for (const href of [
    'https://external.example/v2.3.0/zh/user-guide/usage/?keep=1#keep',
    'https://docs.example:8443/v2.3.0/ja/overview/?keep=1#keep'
]) {
    const external = new Element(href);
    events.click({target: external});
    assert.equal(external.href, href, 'another origin keeps its own URL');
}

const ordinaryHref = 'https://docs.example/v2.3.0/overview/?keep=1#keep';
const ordinary = new Element(ordinaryHref, false);
events.click({target: ordinary});
assert.equal(ordinary.href, ordinaryHref, 'ordinary links are untouched');
assert.doesNotThrow(() => events.click({target: {}}));
"""
