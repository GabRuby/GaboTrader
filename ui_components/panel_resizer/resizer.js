(function () {
  const DIVIDER_WIDTH = 16;
  const divider = document.getElementById("divider");
  let initialRatio = 0.19;
  let initialWidth = null;
  let dragPointerId = null;
  let preferredPanelWidth = null;
  let observedColumns = null;
  let lastAvailableWidth = null;

  function post(type, payload) {
    window.parent.postMessage(
      Object.assign(
        { isStreamlitMessage: true, type: type },
        payload || {}
      ),
      "*"
    );
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function currentColumns() {
    const parent = window.parent;
    const columns = window.frameElement.closest(
      '[data-testid="stHorizontalBlock"]'
    );
    if (!columns || columns.children.length < 3) return null;

    columns.style.flexWrap = "nowrap";
    const dividerColumn = columns.children[1];
    dividerColumn.style.flex = "0 0 " + DIVIDER_WIDTH + "px";
    dividerColumn.style.width = DIVIDER_WIDTH + "px";
    dividerColumn.style.maxWidth = DIVIDER_WIDTH + "px";
    const columnGap = parseFloat(parent.getComputedStyle(columns).columnGap) || 0;
    const dividerWidth = dividerColumn.getBoundingClientRect().width;
    return {
      columns: columns,
      columnGap: columnGap,
      dividerWidth: dividerWidth,
      availableWidth: Math.max(
        1,
        columns.clientWidth - columnGap * 2 - dividerWidth
      ),
      viewportWidth: parent.innerWidth,
    };
  }

  function boundedWidth(requestedWidth, dimensions) {
    const minimumCenter = Math.min(560, dimensions.availableWidth * 0.5);
    const minimumPanel = dimensions.availableWidth * 0.125;
    const maximumPanel = Math.max(
      minimumPanel,
      Math.min(
        dimensions.availableWidth * 0.375,
        dimensions.viewportWidth * 0.375,
        dimensions.availableWidth - minimumCenter
      )
    );
    return clamp(requestedWidth, minimumPanel, maximumPanel);
  }

  function applyPanelWidth(requestedWidth, dimensions) {
    const panelWidth = boundedWidth(requestedWidth, dimensions);
    const centerWidth = dimensions.availableWidth - panelWidth;
    const children = dimensions.columns.children;
    children[0].style.flex = "0 0 " + centerWidth + "px";
    children[0].style.maxWidth = centerWidth + "px";
    children[2].style.flex = "0 0 " + panelWidth + "px";
    children[2].style.maxWidth = panelWidth + "px";
    return panelWidth;
  }

  function widthAtPointer(clientX, dimensions) {
    const frameLeft = window.frameElement.getBoundingClientRect().left;
    const pointerX = frameLeft + clientX;
    const rightEdge = dimensions.columns.getBoundingClientRect().right;
    return boundedWidth(
      rightEdge - pointerX
        - dimensions.columnGap
        - dimensions.dividerWidth / 2,
      dimensions
    );
  }

  function reportWidth(panelWidth, dimensions) {
    post("streamlit:setComponentValue", {
      value: {
        ratio: panelWidth / dimensions.availableWidth,
        width_px: panelWidth,
      },
      dataType: "json",
    });
  }

  function updatePreview(clientX) {
    const dimensions = currentColumns();
    if (!dimensions) return initialRatio;
    preferredPanelWidth = applyPanelWidth(
      widthAtPointer(clientX, dimensions),
      dimensions
    );
    const ratio = preferredPanelWidth / dimensions.availableWidth;
    divider.setAttribute("aria-valuenow", String(Math.round(ratio * 100)));
    return ratio;
  }

  function syncToPanelResize() {
    const dimensions = currentColumns();
    if (!dimensions || dragPointerId !== null) return;
    if (
      lastAvailableWidth !== null
      && Math.abs(dimensions.availableWidth - lastAvailableWidth) < 1
    ) return;
    lastAvailableWidth = dimensions.availableWidth;
    preferredPanelWidth = applyPanelWidth(
      preferredPanelWidth === null
        ? (initialWidth || initialRatio * dimensions.availableWidth)
        : preferredPanelWidth,
      dimensions
    );
  }

  const resizeObserver = new ResizeObserver(syncToPanelResize);

  function resizeWorkspaceAreas() {
    const dimensions = currentColumns();
    if (!dimensions) return;
    const layout = window.frameElement.closest(".st-key-workspace-layout");
    if (!layout) return;

    const height = Math.max(
      120,
      window.parent.innerHeight - layout.getBoundingClientRect().top - 12
    );
    for (const [index, column] of Array.from(dimensions.columns.children).entries()) {
      column.style.height = height + "px";
      column.style.maxHeight = height + "px";
      column.style.minWidth = index === 1 ? DIVIDER_WIDTH + "px" : "0";
      column.style.minHeight = "0";
      column.style.overflowX = index === 1 ? "visible" : "hidden";
      column.style.overflowY = index === 1 ? "hidden" : "auto";
      column.style.overscrollBehavior = "contain";
    }
    post("streamlit:setFrameHeight", { height: height });
  }

  function observeLayout() {
    const dimensions = currentColumns();
    if (!dimensions) return;
    if (observedColumns !== dimensions.columns) {
      if (observedColumns) resizeObserver.unobserve(observedColumns);
      observedColumns = dimensions.columns;
      resizeObserver.observe(observedColumns);
    }
    lastAvailableWidth = dimensions.availableWidth;
    preferredPanelWidth = applyPanelWidth(
      initialWidth || initialRatio * dimensions.availableWidth,
      dimensions
    );
    resizeWorkspaceAreas();
  }

  function sendRatio(clientX) {
    updatePreview(clientX);
    const dimensions = currentColumns();
    if (dimensions && preferredPanelWidth !== null) {
      reportWidth(preferredPanelWidth, dimensions);
    }
  }

  divider.addEventListener("pointerdown", function (event) {
    if (event.button !== 0 && event.pointerType !== "touch") return;
    dragPointerId = event.pointerId;
    divider.classList.add("dragging");
    divider.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  divider.addEventListener("pointermove", function (event) {
    if (event.pointerId !== dragPointerId) return;
    updatePreview(event.clientX);
  });

  function finishDrag(event) {
    if (event.pointerId !== dragPointerId) return;
    sendRatio(event.clientX);
    dragPointerId = null;
    divider.classList.remove("dragging");
  }

  divider.addEventListener("pointerup", finishDrag);
  divider.addEventListener("pointercancel", finishDrag);

  divider.addEventListener("keydown", function (event) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const centerX = window.innerWidth / 2;
    sendRatio(centerX + (event.key === "ArrowLeft" ? -24 : 24));
  });

  window.addEventListener("message", function (event) {
    if (event.data && event.data.type === "streamlit:render") {
      initialRatio = Number(event.data.args.initial_ratio) || initialRatio;
      const nextWidth = Number(event.data.args.initial_width);
      initialWidth = Number.isFinite(nextWidth) && nextWidth > 0
        ? nextWidth
        : null;
      divider.setAttribute(
        "aria-valuenow",
        String(Math.round(initialRatio * 100))
      );
      observeLayout();
    }
  });

  window.parent.addEventListener("resize", resizeWorkspaceAreas);
  post("streamlit:componentReady", { apiVersion: 1 });
  post("streamlit:setFrameHeight", { height: 480 });
})();
