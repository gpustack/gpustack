const fs = require("fs"), path = require("path"), vm = require("vm");
const root = process.argv[1], worker = process.argv[2];
let listener, reply;
const context = vm.createContext({
  console,
  addEventListener: (type, handler) => { listener = handler; },
  postMessage: value => { reply = value; }
});
context.self = context;
context.importScripts = (...urls) => {
  for (const url of urls)
    vm.runInContext(fs.readFileSync(path.resolve(path.dirname(worker), url), "utf8"), context);
};
vm.runInContext(fs.readFileSync(worker, "utf8"), context);
(async () => {
  const data = JSON.parse(fs.readFileSync(path.join(root, "search/search_index.json")));
  data.options = { suggest: false };
  await listener({ data: { type: 0, data } });
  const results = {};
  for (const query of JSON.parse(process.argv[3])) {
    await listener({ data: { type: 2, data: query } });
    results[query] = reply.data.items.flat().map(item => item.location);
  }
  process.stdout.write(JSON.stringify(results));
})();
