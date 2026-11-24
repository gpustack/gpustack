(() => {
  function updateSearch(event) {
    const input = event.target;
    if (!event.isComposing && input.matches('[data-md-component="search-query"]')) {
      // Material reads the query on keyup. Notify it after paste or IME commit.
      input.dispatchEvent(new Event("keyup"));
    }
  }

  document.addEventListener("input", updateSearch);
  document.addEventListener("compositionend", updateSearch);
})();
