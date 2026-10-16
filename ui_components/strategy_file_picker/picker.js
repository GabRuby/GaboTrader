(function () {
  const root = document.getElementById("root");
  let currentArgs = {};

  function post(type, payload) {
    window.parent.postMessage(
      Object.assign({ isStreamlitMessage: true, type: type }, payload || {}),
      "*"
    );
  }

  function renderDropzone() {
    root.innerHTML =
      '<div class="dropzone" id="dropzone">' +
      "<strong>Arrastra un archivo .py aquí</strong>" +
      "<span>Solo se analizan archivos .py de estrategias</span>" +
      "</div>";
    const dropzone = document.getElementById("dropzone");
    ["dragenter", "dragover"].forEach(function (eventName) {
      dropzone.addEventListener(eventName, function (event) {
        event.preventDefault();
        dropzone.classList.add("dragging");
      });
    });
    ["dragleave", "drop"].forEach(function (eventName) {
      dropzone.addEventListener(eventName, function (event) {
        event.preventDefault();
        dropzone.classList.remove("dragging");
      });
    });
    dropzone.addEventListener("drop", function (event) {
      const files = event.dataTransfer && event.dataTransfer.files;
      if (files && files.length) readFile(files[0]);
    });
  }

  function renderActions() {
    root.innerHTML =
      '<input id="file-input" type="file" accept=".py,text/x-python">' +
      '<div class="actions">' +
      '<button id="search-button" type="button">Buscar estrategia</button>' +
      '<button id="cancel-button" type="button">Cancelar</button>' +
      '<button id="load-button" class="primary" type="button"></button>' +
      "</div>";

    const input = document.getElementById("file-input");
    document.getElementById("search-button").addEventListener("click", function () {
      input.value = "";
      input.click();
    });
    input.addEventListener("change", function () {
      if (input.files && input.files.length) readFile(input.files[0]);
    });
    document.getElementById("cancel-button").addEventListener("click", function () {
      sendEvent({ action: "cancel" });
    });
    const loadButton = document.getElementById("load-button");
    loadButton.textContent = currentArgs.load_label || "Cargar en Editor";
    loadButton.disabled = currentArgs.can_load !== true;
    loadButton.title = currentArgs.load_help || "";
    loadButton.addEventListener("click", function () {
      if (currentArgs.can_load === true) sendEvent({ action: "load" });
    });
  }

  function sendEvent(event) {
    event.event_id =
      Date.now().toString(36) + "-" + Math.random().toString(36).slice(2);
    post("streamlit:setComponentValue", { value: event, dataType: "json" });
  }

  function readFile(file) {
    const reader = new FileReader();
    reader.onerror = function () {
      sendEvent({
        action: "file_selected",
        name: file.name,
        read_error: "El navegador no pudo leer el archivo seleccionado.",
      });
    };
    reader.onload = function () {
      const bytes = new Uint8Array(reader.result);
      let binary = "";
      for (let offset = 0; offset < bytes.length; offset += 0x8000) {
        binary += String.fromCharCode.apply(
          null,
          bytes.subarray(offset, Math.min(offset + 0x8000, bytes.length))
        );
      }
      sendEvent({
        action: "file_selected",
        name: file.name,
        content_base64: btoa(binary),
      });
    };
    reader.readAsArrayBuffer(file);
  }

  window.addEventListener("message", function (event) {
    if (!event.data || event.data.type !== "streamlit:render") return;
    currentArgs = event.data.args || {};
    if (currentArgs.part === "actions") renderActions();
    else renderDropzone();
    post("streamlit:setFrameHeight", {
      height: currentArgs.part === "actions" ? 58 : 86,
    });
  });

  post("streamlit:componentReady", { apiVersion: 1 });
  post("streamlit:setFrameHeight", { height: 86 });
})();
