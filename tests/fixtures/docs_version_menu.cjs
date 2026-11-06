// Run Material's own version menu and sitemap navigation against local mike output.
const fs = require("fs"), path = require("path"), vm = require("vm");
const [published, language, targetVersion = "2.2", scenario = "{}"] = process.argv.slice(2);
const options = JSON.parse(scenario);
const prefix = language === "en" ? "" : `${language}/`;
const current = `/docs/latest/${prefix}user-guide/guide/`;
const location = { href: `https://docs.example.test${current}?q=language#cluster-management` };
const html = fs.readFileSync(path.join(published, "latest", prefix, "user-guide/guide/index.html"), "utf8");
const configuration = JSON.parse(html.match(/<script id="__config"[^>]*>(.*?)<\/script>/s)[1]);
const asset = html.match(/<script src="([^"]*assets\/javascripts\/bundle[^\"]+\.js)"/)[1];
const bundlePath = path.join(published, new URL(asset, location.href).pathname.replace("/docs/", ""));
let bundle = fs.readFileSync(bundlePath, "utf8");
const fetchPosition = bundle.indexOf('"../versions.json"');
const functionStart = bundle.lastIndexOf("function ", fetchPosition);
const initialize = bundle.slice(functionStart).match(/^function ([\w$]+)\(/)[1];
const bootstrap = bundle.indexOf('document.documentElement.classList.remove("no-js")');
if (bootstrap < 0) throw new Error("Unsupported Material bootstrap in test fixture");
bundle = bundle.slice(0, bootstrap) + `globalThis.initializeVersionMenu=${initialize};})();`;
if (language !== "en") bundle += "})();";

class Element {
  constructor(tag) { this.tagName = tag; this.attributes = {}; this.children = []; this.listeners = {}; this.innerHTML = ""; }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key] || null; }
  appendChild(child) { this.children.push(child); child.parentElement = this; return child; }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  removeEventListener(type, handler) { this.listeners[type] = (this.listeners[type] || []).filter(item => item !== handler); }
  get href() { return new URL(this.attributes.href || "", location.href).href; }
  get target() { return this.attributes.target || ""; }
  closest(selector) {
    if (selector === "a" && this.tagName === "a") return this;
    if (selector === ".md-version" && this.attributes.class === "md-version") return this;
    return this.parentElement?.closest(selector) || null;
  }
  querySelector(selector) { return selector === "loc" ? { textContent: this.textContent } : null; }
  querySelectorAll() { return []; }
}
const body = new Element("body"), topic = new Element("div");
const document = {
  body,
  nodeType: 9,
  documentElement: { classList: { add() {} }, setAttribute() {} },
  addEventListener() {}, removeEventListener() {},
  createElement: tag => new Element(tag),
  querySelector: selector => selector === "#__config" ? { textContent: JSON.stringify(configuration) } : selector === ".md-header__topic" ? topic : new Element("input"),
  querySelectorAll: () => []
};
const requests = [], abortedRequests = [], lateLoads = [], activeRequests = new Set(), activeTimers = new Set();
let expiredTimers = 0, clearedTimers = 0;
class XMLHttpRequest {
  constructor() { this.listeners = {}; }
  open(method, url) { this.url = url; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  abort() {
    if (!activeRequests.delete(this)) return;
    this.aborted = true;
    abortedRequests.push(this.url);
    this.listeners.abort();
  }
  send() {
    requests.push(this.url);
    activeRequests.add(this);
    const relative = new URL(this.url).pathname.replace("/docs/", "");
    const filename = path.join(published, relative);
    this.status = (options.failedRequests || []).includes(relative) ? 503 : fs.existsSync(filename) ? 200 : 404;
    this.response = { text: async () => {
      let text = fs.readFileSync(filename, "utf8");
      if ((options.omitCurrentPageFrom || []).includes(relative)) {
        text = text.replace(/<url>\s*<loc>[^<]*\/user-guide\/guide\/<\/loc>[\s\S]*?<\/url>/g, "");
      }
      return text;
    } };
    if ((options.hangRequests || []).includes(relative)) return;
    const deliver = () => {
      activeRequests.delete(this);
      if (this.aborted) lateLoads.push(this.url);
      this.listeners.load();
    };
    const delay = (options.delayedRequests || {})[relative];
    if (delay) setTimeout(deliver, delay);
    else queueMicrotask(deliver);
  }
}
class DOMParser {
  parseFromString(text) {
    return {
      querySelectorAll: selector => selector === "url" ? [...text.matchAll(/<loc>(.*?)<\/loc>/g)].map(match => {
        const element = new Element("url"); element.textContent = match[1]; return element;
      }) : []
    };
  }
}
const context = vm.createContext({
  URL, console, setInterval, clearInterval,
  setTimeout(callback, delay, ...args) {
    const timer = setTimeout(() => { activeTimers.delete(timer); expiredTimers++; callback(...args); }, delay * (options.timerScale || 1));
    activeTimers.add(timer);
    return timer;
  },
  clearTimeout(timer) { if (activeTimers.delete(timer)) clearedTimers++; clearTimeout(timer); },
  location, document, Element, Node: Element, XMLHttpRequest, DOMParser,
  navigator: { userAgent: "Documentation regression" },
  CustomEvent: class {}, dispatchEvent() {}, addEventListener() {}, removeEventListener() {},
  sessionStorage: {}, __md_get: () => null, __md_set: () => {}
});
context.window = context;
try { vm.runInContext(bundle, context); }
catch (error) { throw new Error(error.message); }
const settle = (delay = options.waitMs || 20) => new Promise(resolve => setTimeout(resolve, delay));
const descendants = element => [element, ...element.children.flatMap(descendants)];
(async () => {
  context.initializeVersionMenu({ document$: { pipe: () => ({ subscribe() {} }) } });
  await settle();
  const menu = topic.children.find(element => element.attributes.class === "md-version");
  if (!menu) throw new Error(`No version menu; requests: ${requests.join(", ")}`);
  const links = descendants(menu).filter(element => element.tagName === "a");
  const target = links.find(element => new URL(element.href).pathname.startsWith(`/docs/${targetVersion}/`));
  if (!target) throw new Error("The older version is absent from Material's menu");
  const targetHref = target.href;
  if (options.waitAfterMenu) await settle(options.waitAfterMenu);
  let prevented = false;
  const event = { target, metaKey: options.modifier === "meta", ctrlKey: options.modifier === "ctrl", preventDefault() { prevented = true; } };
  for (const handler of body.listeners.click || []) handler(event);
  await settle();
  process.stdout.write(JSON.stringify({ requests, abortedRequests, lateLoads, expiredTimers, clearedTimers, activeTimers: activeTimers.size,
    activeRequests: [...activeRequests].map(request => request.url),
    targets: links.map(element => element.href), targetHref, navigated: location.href, prevented,
    selected: descendants(menu).find(element => element.attributes.class === "md-version__current").innerHTML }));
})();
