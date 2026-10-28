(function () {
  const FALLBACK_MESSAGE = "地图服务暂不可用，请检查高德地图配置";

  function toLatLng(value) {
    if (!value) {
      return null;
    }
    if (Array.isArray(value)) {
      return { lat: Number(value[0]), lng: Number(value[1]) };
    }
    if (typeof value.getLat === "function" && typeof value.getLng === "function") {
      return { lat: Number(value.getLat()), lng: Number(value.getLng()) };
    }
    return { lat: Number(value.lat), lng: Number(value.lng) };
  }

  function toAmapLngLat(value) {
    const latlng = toLatLng(value);
    if (!latlng || !Number.isFinite(latlng.lat) || !Number.isFinite(latlng.lng)) {
      return null;
    }
    return [latlng.lng, latlng.lat];
  }

  function makeLatLng(lat, lng) {
    return { lat: Number(lat), lng: Number(lng) };
  }

  function finiteNumber(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  class LatLngBounds {
    constructor(points = []) {
      this.points = [];
      points.forEach(point => this.extend(point));
    }

    extend(point) {
      const latlng = toLatLng(point);
      if (latlng && Number.isFinite(latlng.lat) && Number.isFinite(latlng.lng)) {
        this.points.push(latlng);
      }
      return this;
    }

    isValid() {
      return this.points.length > 0;
    }

    pad(ratio = 0) {
      if (!this.isValid()) {
        return new LatLngBounds();
      }
      const { minLat, maxLat, minLng, maxLng } = this._range();
      const latPad = Math.max(0.02, (maxLat - minLat) * ratio);
      const lngPad = Math.max(0.02, (maxLng - minLng) * ratio);
      return new LatLngBounds([
        [minLat - latPad, minLng - lngPad],
        [maxLat + latPad, maxLng + lngPad]
      ]);
    }

    toAmapBounds(AMap) {
      if (!this.isValid() || !AMap) {
        return null;
      }
      const { minLat, maxLat, minLng, maxLng } = this._range();
      return new AMap.Bounds(
        new AMap.LngLat(minLng, minLat),
        new AMap.LngLat(maxLng, maxLat)
      );
    }

    _range() {
      const lats = this.points.map(point => point.lat);
      const lngs = this.points.map(point => point.lng);
      return {
        minLat: Math.min(...lats),
        maxLat: Math.max(...lats),
        minLng: Math.min(...lngs),
        maxLng: Math.max(...lngs)
      };
    }
  }

  class AmapMapAdapter {
    constructor(containerId, options = {}) {
      this.container = document.getElementById(containerId);
      this.AMap = window.AMap || null;
      this.layers = new Set();
      this.ready = false;
      this.panes = {};
      this.available = !!(this.container && this.AMap);
      const minZoom = Math.max(2, finiteNumber(options.minZoom, 3));
      const maxZoom = Math.max(20, finiteNumber(options.maxZoom, 20));
      this.minZoom = Math.min(minZoom, maxZoom);
      this.maxZoom = Math.max(maxZoom, this.minZoom);
      this.interactionStatus = {
        zoomEnable: true,
        dragEnable: true,
        scrollWheel: true,
        doubleClickZoom: true,
        keyboardEnable: true,
        touchZoom: true,
        jogEnable: true,
        resizeEnable: true
      };
      if (!this.available) {
        this._showUnavailable();
        return;
      }

      this.container.classList.remove("map-service-unavailable");
      if (!this.container.hasAttribute("tabindex")) {
        this.container.setAttribute("tabindex", "0");
      }
      try {
        this.map = new this.AMap.Map(this.container, {
          center: toAmapLngLat(options.center) || [104.1954, 35.8617],
          zoom: Math.max(this.minZoom, Math.min(this.maxZoom, finiteNumber(options.zoom, 4))),
          zooms: [this.minZoom, this.maxZoom],
          viewMode: "2D",
          ...this.interactionStatus,
          showLabel: true,
          mapStyle: "amap://styles/normal",
          features: ["bg", "road", "point"]
        });
      } catch (_error) {
        this.available = false;
        this._showUnavailable();
        return;
      }
      this.map.on("complete", () => {
        this.ready = true;
        this._enableMapInteractions();
        this.invalidateSize();
      });
      this._enableMapInteractions();
      this._installWheelIsolation();
      this._installControls();
    }

    _enableMapInteractions() {
      if (!this.map) {
        return;
      }
      if (typeof this.map.setStatus === "function") {
        this.map.setStatus({ ...this.interactionStatus });
      }
      if (typeof this.map.setZooms === "function") {
        this.map.setZooms([this.minZoom, this.maxZoom]);
      }
    }

    _installWheelIsolation() {
      if (!this.container || this._wheelIsolationInstalled) {
        return;
      }
      this._wheelIsolationInstalled = true;
      this.container.addEventListener("wheel", event => {
        const interactiveOverlay = this._isInteractiveOverlayTarget(event.target);
        event.stopPropagation();
        if (interactiveOverlay) {
          return;
        }
        const beforeZoom = finiteNumber(this.map?.getZoom?.(), this.minZoom);
        const direction = event.deltaY < 0 ? 1 : -1;
        window.setTimeout(() => this._fallbackStepZoom(direction, beforeZoom), 40);
        event.preventDefault();
      }, { passive: false });

      this.container.addEventListener("dblclick", event => {
        if (this._isInteractiveOverlayTarget(event.target)) {
          return;
        }
        const beforeZoom = finiteNumber(this.map?.getZoom?.(), this.minZoom);
        window.setTimeout(() => this._fallbackStepZoom(1, beforeZoom), 40);
      });
    }

    _isInteractiveOverlayTarget(target) {
      return !!target?.closest?.(
        ".route-place-detail-popup, .amap-paper-info-window, .amap-info-content"
      );
    }

    _fallbackStepZoom(direction, beforeZoom) {
      if (!this.map || typeof this.map.getZoom !== "function" || typeof this.map.setZoom !== "function") {
        return;
      }
      const currentZoom = finiteNumber(this.map.getZoom(), beforeZoom);
      if (Math.abs(currentZoom - beforeZoom) > 0.01) {
        return;
      }
      const nextZoom = Math.max(this.minZoom, Math.min(this.maxZoom, beforeZoom + direction));
      if (Math.abs(nextZoom - beforeZoom) > 0.01) {
        this.map.setZoom(nextZoom);
      }
    }

    _installControls() {
      if (!this.AMap || !this.map || typeof this.AMap.plugin !== "function") {
        return;
      }
      this.AMap.plugin(["AMap.ToolBar", "AMap.Scale"], () => {
        if (this.AMap.ToolBar) {
          this.map.addControl(new this.AMap.ToolBar({
            position: "RB",
            liteStyle: false
          }));
        }
        if (this.AMap.Scale) {
          this.map.addControl(new this.AMap.Scale());
        }
      });
    }

    _showUnavailable() {
      if (!this.container) {
        return;
      }
      this.container.classList.add("map-service-unavailable");
      if (this.container.querySelector(".map-service-fallback")) {
        return;
      }
      const fallback = document.createElement("div");
      fallback.className = "map-service-fallback";
      fallback.textContent = FALLBACK_MESSAGE;
      this.container.appendChild(fallback);
    }

    getContainer() {
      return this.container;
    }

    createPane(name) {
      if (!this.container) {
        return null;
      }
      if (this.panes[name]) {
        return this.panes[name];
      }
      const pane = document.createElement("div");
      pane.className = `amap-adapter-pane ${name}`;
      pane.style.position = "absolute";
      pane.style.inset = "0";
      pane.style.pointerEvents = "none";
      this.container.appendChild(pane);
      this.panes[name] = pane;
      return pane;
    }

    getPane(name) {
      return this.panes[name] || null;
    }

    getPanes() {
      return { overlayPane: this.container, ...this.panes };
    }

    on(events, handler) {
      if (!this.map || typeof handler !== "function") {
        return this;
      }
      String(events || "").split(/\s+/).filter(Boolean).forEach(eventName => {
        this.map.on(eventName, handler);
      });
      return this;
    }

    once(eventName, handler) {
      if (!this.map || typeof handler !== "function") {
        return this;
      }
      const wrapped = event => {
        this.map.off(eventName, wrapped);
        handler(event);
      };
      this.map.on(eventName, wrapped);
      return this;
    }

    whenReady(callback) {
      if (typeof callback !== "function") {
        return;
      }
      window.setTimeout(callback, this.available ? 0 : 1);
    }

    invalidateSize() {
      this.map?.resize?.();
    }

    addLayer(layer) {
      if (!layer) {
        return this;
      }
      if (typeof layer._addToMap === "function") {
        layer._addToMap(this);
      }
      this.layers.add(layer);
      return this;
    }

    removeLayer(layer) {
      if (!layer) {
        return this;
      }
      if (typeof layer._removeFromMap === "function") {
        layer._removeFromMap();
      }
      this.layers.delete(layer);
      return this;
    }

    hasLayer(layer) {
      return !!layer && (this.layers.has(layer) || layer.__map === this);
    }

    fitBounds(boundsLike, options = {}) {
      if (!this.map || !this.AMap) {
        return;
      }
      const bounds = boundsLike instanceof LatLngBounds
        ? boundsLike
        : new LatLngBounds(Array.isArray(boundsLike) ? boundsLike : []);
      const amapBounds = bounds.toAmapBounds(this.AMap);
      if (!amapBounds) {
        return;
      }
      const padding = Array.isArray(options.padding)
        ? [options.padding[1] || 12, options.padding[0] || 12, options.padding[1] || 12, options.padding[0] || 12]
        : [18, 18, 18, 18];
      if (typeof this.map.setBounds === "function") {
        this.map.setBounds(amapBounds, false, padding);
      } else {
        this.map.setFitView(null, false, padding);
      }
      if (Number.isFinite(Number(options.maxZoom))) {
        window.setTimeout(() => {
          if (this.map.getZoom() > Number(options.maxZoom)) {
            this.map.setZoom(Number(options.maxZoom));
          }
        }, 0);
      }
    }

    flyTo(latlng, zoom) {
      if (!this.map) {
        return;
      }
      const center = toAmapLngLat(latlng);
      if (!center) {
        return;
      }
      this.map.setZoomAndCenter(Number(zoom) || this.map.getZoom(), center, false);
    }

    latLngToContainerPoint(latlng) {
      if (!this.map) {
        return { x: 0, y: 0 };
      }
      const lngLat = toAmapLngLat(latlng);
      if (!lngLat) {
        return { x: 0, y: 0 };
      }
      const pixel = this.map.lngLatToContainer(lngLat);
      return {
        x: typeof pixel.getX === "function" ? pixel.getX() : Number(pixel.x) || 0,
        y: typeof pixel.getY === "function" ? pixel.getY() : Number(pixel.y) || 0
      };
    }

    openInfoWindow(html, latlng) {
      if (!this.map || !this.AMap) {
        return;
      }
      const position = toAmapLngLat(latlng);
      if (!position) {
        return;
      }
      const info = new this.AMap.InfoWindow({
        isCustom: true,
        autoMove: true,
        offset: new this.AMap.Pixel(0, -10),
        content: `<div class="amap-paper-info-window">${html}</div>`
      });
      info.open(this.map, position);
    }
  }

  class LayerGroup {
    constructor(layers = []) {
      this.layers = [];
      this.__map = null;
      layers.forEach(layer => this.addLayer(layer));
    }

    addLayer(layer) {
      if (!layer) {
        return this;
      }
      this.layers.push(layer);
      if (this.__map) {
        layer.addTo(this.__map);
      }
      return this;
    }

    clearLayers() {
      this.layers.forEach(layer => layer?._removeFromMap?.());
      this.layers = [];
      return this;
    }

    addTo(target) {
      if (target instanceof AmapMapAdapter) {
        target.addLayer(this);
      }
      return this;
    }

    _addToMap(map) {
      this.__map = map;
      this.layers.forEach(layer => layer.addTo(map));
    }

    _removeFromMap() {
      this.layers.forEach(layer => layer?._removeFromMap?.());
      this.__map = null;
    }

    bringToFront() {
      this.layers.forEach(layer => layer?.bringToFront?.());
      return this;
    }

    eachLayer(callback) {
      this.layers.forEach(callback);
    }
  }

  class AmapOverlayLayer {
    constructor() {
      this.__map = null;
      this.overlay = null;
      this.popupHtml = "";
      this.latlng = null;
    }

    addTo(target) {
      if (target instanceof LayerGroup) {
        target.addLayer(this);
      } else if (target instanceof AmapMapAdapter) {
        target.addLayer(this);
      }
      return this;
    }

    _addToMap(map) {
      this.__map = map;
      if (!map.available) {
        return;
      }
      if (!this.overlay) {
        this.overlay = this._createOverlay(map);
        this._flushPendingEvents();
      }
      this.overlay?.setMap?.(map.map);
    }

    _removeFromMap() {
      this.overlay?.setMap?.(null);
      this.__map = null;
    }

    on(eventName, handler) {
      if (!this.overlay || !this.__map?.AMap || typeof handler !== "function") {
        this._pendingEvents = this._pendingEvents || [];
        this._pendingEvents.push([eventName, handler]);
        return this;
      }
      this.__map.AMap.Event.addListener(this.overlay, eventName, event => {
        handler({ ...event, latlng: event?.lnglat ? toLatLng(event.lnglat) : this.latlng });
      });
      return this;
    }

    bindPopup(html) {
      this.popupHtml = html || "";
      this.on("click", event => {
        this.__map?.openInfoWindow(this.popupHtml, event?.latlng || this.latlng);
      });
      return this;
    }

    bindTooltip() {
      return this;
    }

    openPopup() {
      this.__map?.openInfoWindow(this.popupHtml, this.latlng);
      return this;
    }

    bringToFront() {
      this.overlay?.setzIndex?.(999);
      this.overlay?.setzIndex?.(999);
      return this;
    }

    _flushPendingEvents() {
      const pending = this._pendingEvents || [];
      this._pendingEvents = [];
      pending.forEach(([eventName, handler]) => this.on(eventName, handler));
    }
  }

  class CircleMarkerLayer extends AmapOverlayLayer {
    constructor(latlng, options = {}) {
      super();
      this.latlng = toLatLng(latlng);
      this.options = options;
    }

    _createOverlay(map) {
      const center = toAmapLngLat(this.latlng);
      const overlay = new map.AMap.CircleMarker({
        center,
        radius: Number(this.options.radius) || 8,
        strokeColor: this.options.color || "#1f6b56",
        strokeWeight: Number(this.options.weight) || 0,
        strokeOpacity: Number(this.options.opacity ?? 1),
        fillColor: this.options.fillColor || this.options.color || "#1f6b56",
        fillOpacity: Number(this.options.fillOpacity ?? 0.9),
        bubble: true,
        zIndex: 90
      });
      return overlay;
    }

    setLatLng(latlng) {
      this.latlng = toLatLng(latlng);
      const center = toAmapLngLat(this.latlng);
      if (center && this.overlay?.setCenter) {
        this.overlay.setCenter(center);
      }
      return this;
    }

    setStyle(style = {}) {
      if (!this.overlay) {
        this.options = { ...this.options, ...style };
        return this;
      }
      if (style.fillOpacity !== undefined) {
        this.overlay.setOptions({ fillOpacity: Number(style.fillOpacity) });
      }
      if (style.opacity !== undefined) {
        this.overlay.setOptions({ strokeOpacity: Number(style.opacity) });
      }
      return this;
    }
  }

  class MarkerLayer extends AmapOverlayLayer {
    constructor(latlng, options = {}) {
      super();
      this.latlng = toLatLng(latlng);
      this.options = options;
      this.element = null;
    }

    _createOverlay(map) {
      const content = document.createElement("div");
      content.className = this.options.icon?.className || "";
      content.innerHTML = this.options.icon?.html || "";
      this.element = content;
      const overlay = new map.AMap.Marker({
        position: toAmapLngLat(this.latlng),
        content,
        anchor: "center",
        offset: new map.AMap.Pixel(0, 0),
        bubble: true,
        zIndex: 120
      });
      return overlay;
    }

    getElement() {
      return this.element;
    }

    getLatLng() {
      return this.latlng;
    }
  }

  class PolylineLayer extends AmapOverlayLayer {
    constructor(latlngs = [], options = {}) {
      super();
      this.latlngs = latlngs;
      this.options = options;
    }

    _createOverlay(map) {
      return new map.AMap.Polyline({
        path: this._path(),
        strokeColor: this.options.color || "#b88945",
        strokeWeight: Number(this.options.weight) || 2,
        strokeOpacity: Number(this.options.opacity ?? 0.8),
        strokeStyle: this.options.dashArray ? "dashed" : "solid",
        lineJoin: "round",
        lineCap: "round",
        zIndex: 80
      });
    }

    _path() {
      return (this.latlngs || []).map(toAmapLngLat).filter(Boolean);
    }

    setLatLngs(latlngs) {
      this.latlngs = latlngs || [];
      this.overlay?.setPath?.(this._path());
      return this;
    }
  }

  class PolygonLayer extends AmapOverlayLayer {
    constructor(paths = [], options = {}) {
      super();
      this.paths = paths;
      this.options = options;
    }

    _createOverlay(map) {
      return new map.AMap.Polygon({
        path: this.paths,
        strokeColor: this.options.color || "#1f4f43",
        strokeWeight: Number(this.options.weight) || 2,
        strokeOpacity: Number(this.options.opacity ?? 0.8),
        fillColor: this.options.fillColor || "#ead9ac",
        fillOpacity: Number(this.options.fillOpacity ?? 0.04),
        bubble: false,
        zIndex: 60
      });
    }
  }

  class GeoJsonLayer extends LayerGroup {
    constructor(geojson, options = {}) {
      super();
      this.geojson = geojson;
      this.options = options;
      this._build();
    }

    _build() {
      const features = Array.isArray(this.geojson?.features) ? this.geojson.features : [];
      features.forEach(feature => {
        const geometry = feature?.geometry || {};
        if (geometry.type === "Point") {
          const coords = geometry.coordinates || [];
          const latlng = makeLatLng(coords[1], coords[0]);
          const layer = this.options.pointToLayer
            ? this.options.pointToLayer(feature, latlng)
            : new CircleMarkerLayer(latlng);
          this.options.onEachFeature?.(feature, layer);
          this.addLayer(layer);
          return;
        }
        const style = typeof this.options.style === "function"
          ? this.options.style(feature)
          : (this.options.style || {});
        const polygons = geometry.type === "Polygon"
          ? [geometry.coordinates]
          : geometry.type === "MultiPolygon"
            ? geometry.coordinates
            : [];
        polygons.forEach(polygon => {
          const rings = polygon.map(ring => ring.map(coord => [Number(coord[0]), Number(coord[1])]));
          this.addLayer(new PolygonLayer(rings, style));
        });
      });
    }

    getBounds() {
      const bounds = new LatLngBounds();
      const visit = coords => {
        if (!Array.isArray(coords)) {
          return;
        }
        if (coords.length >= 2 && Number.isFinite(Number(coords[0])) && Number.isFinite(Number(coords[1]))) {
          bounds.extend([Number(coords[1]), Number(coords[0])]);
          return;
        }
        coords.forEach(visit);
      };
      (this.geojson?.features || []).forEach(feature => visit(feature?.geometry?.coordinates));
      return bounds;
    }
  }

  class HeatLayer extends LayerGroup {
    constructor(points = [], options = {}) {
      super();
      points.forEach(point => {
        const intensity = Number(point?.[2]) || 0.3;
        this.addLayer(new CircleMarkerLayer([point[0], point[1]], {
          radius: Math.max(8, Math.min(26, (Number(options.radius) || 20) * (0.35 + intensity * 0.35))),
          stroke: false,
          fillColor: "#d8b36a",
          fillOpacity: Math.max(0.05, Math.min(0.22, intensity * 0.2))
        }));
      });
    }
  }

  function createHexiAmapAdapter(options = {}) {
    const map = new AmapMapAdapter(options.containerId || "map", options);
    const L = {
      layerGroup: layers => new LayerGroup(layers),
      circleMarker: (latlng, layerOptions) => new CircleMarkerLayer(latlng, layerOptions),
      marker: (latlng, layerOptions) => new MarkerLayer(latlng, layerOptions),
      polyline: (latlngs, layerOptions) => new PolylineLayer(latlngs, layerOptions),
      geoJSON: (geojson, layerOptions) => new GeoJsonLayer(geojson, layerOptions),
      heatLayer: (points, layerOptions) => new HeatLayer(points, layerOptions),
      divIcon: options => options || {},
      latLng: makeLatLng,
      latLngBounds: points => new LatLngBounds(points),
      DomEvent: {
        disableClickPropagation: element => element?.addEventListener?.("click", event => event.stopPropagation()),
        disableScrollPropagation: element => element?.addEventListener?.("wheel", event => event.stopPropagation(), { passive: true })
      }
    };
    return { map, L, isAmapAvailable: map.available };
  }

  window.createHexiAmapAdapter = createHexiAmapAdapter;
})();
