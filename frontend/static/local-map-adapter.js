/* Local PPT artwork renderer. No map-provider SDK, tiles, keys or attribution. */
(function () {
  "use strict";
  const W = 1347, H = 987;
  // Pixel centers extracted from the five editable PPT city circles.
  const anchors = [
    { name: "酒泉", lat: 39.7432, lng: 98.5174, x: 304.065422, y: 356.283046 },
    { name: "嘉峪关", lat: 39.7865, lng: 98.2773, x: 626.215312, y: 378.590479 },
    { name: "张掖", lat: 38.9259, lng: 100.4498, x: 778.918153, y: 486.846180 },
    { name: "金昌", lat: 38.5208, lng: 102.1880, x: 1057.534676, y: 545.822489 },
    { name: "武威", lat: 37.9282, lng: 102.6347, x: 1057.534676, y: 618.052925 }
  ];
  function geographicPixel(value) {
    const p = Array.isArray(value) ? { lat: value[0], lng: value[1] } : value;
    let x = (Number(p.lng) - 91.79) / 14.24 * W;
    let y = (42.65 - Number(p.lat)) / 8.9 * H;
    // Artwork city positions are schematic. Snap city coordinates and nearby
    // historical aliases to the original city circle, not a different basemap.
    const nearest = anchors.map(a => ({ a, d: Math.hypot(p.lat - a.lat, p.lng - a.lng) }))
      .sort((a, b) => a.d - b.d)[0];
    if (nearest.d < 0.12) {
      const a = nearest.a;
      const blend = nearest.d <= 0.035 ? 1 : (0.12 - nearest.d) / 0.085;
      x += blend * (a.x - (a.lng - 91.79) / 14.24 * W);
      y += blend * (a.y - (42.65 - a.lat) / 8.9 * H);
    }
    return { x, y };
  }
  const cityAliases = {
        酒泉: "酒泉", 酒泉市: "酒泉", 肃州: "酒泉", PLC_JIUQUAN: "酒泉",
        嘉峪关: "嘉峪关", 嘉峪关市: "嘉峪关", PLC_JIAYUGUAN: "嘉峪关",
        张掖: "张掖", 张掖市: "张掖", 甘州: "张掖", PLC_GANZHOU: "张掖",
        金昌: "金昌", 金昌市: "金昌", PLC_JINCHANG: "金昌",
        武威: "武威", 武威市: "武威", 凉州: "武威", PLC_WUWEI: "武威", PLC_LIANGZHOU: "武威"
      };
  function predictionPixel(value, placeName, margin = 24) {
    const city = anchors.find(a => a.name === cityAliases[String(placeName || "").trim()]);
    if (city) return { x: city.x, y: city.y, kind: "exact", city: city.name };
    const p = Array.isArray(value) ? { lat: value[0], lng: value[1] } : value;
    const actual = { x: (Number(p.lng)-91.79)/14.24*W, y: (42.65-Number(p.lat))/8.9*H };
    return { ...window.HEXI_CORRIDOR_GEOMETRY.constrain(actual, margin), kind: "approximate" };
  }
  window.HEXI_ARTWORK = { width: W, height: H, anchors, geographicPixel, predictionPixel };
  window.createHexiLocalMapAdapter = function ({ containerId = "map" } = {}) {
    const L = window.L;
    // Use one pixel coordinate space for artwork, markers, paths and popups.
    // A facade converts existing geographic calls; Leaflet itself stays in pixels.
    const raw = L.map(containerId, {
      crs: L.CRS.Simple, attributionControl: false, zoomControl: false,
      minZoom: -5, maxZoom: 4, zoomSnap: 0, zoomDelta: 0.5,
      zoomAnimation: false, markerZoomAnimation: false, fadeAnimation: false,
      maxBoundsViscosity: 1, preferCanvas: false
    });
    const imageBounds = L.latLngBounds([[-H, 0], [0, W]]);
    const artwork = L.imageOverlay('/static/ppt-terrain-cities.svg', imageBounds, {
      interactive: false, alt: 'PPT 原始河西走廊地形与城市地图（保留范围边界，无指北针和比例尺）'
    }).addTo(raw);
    raw.setMaxBounds(imageBounds);
    L.control.zoom({ position: 'bottomright' }).addTo(raw);
    const toPixelLatLng = p => { const v = geographicPixel(p); return L.latLng(-v.y, v.x); };
    let predictionFocus = null;
    function focusedPosition(value) {
      const p = L.latLng(value), active = predictionFocus?.geographic();
      return active && Math.hypot(p.lat - active.lat, p.lng - active.lng) < 0.000001
        ? predictionFocus.layer.getLatLng() : toPixelLatLng(value);
    }
    let lastSize = '';
    function coverZoom() {
      const size = raw.getSize();
      return size.x > 0 && size.y > 0
        ? Math.log2(Math.max(size.x / W, size.y / H)) : -1;
    }
    function home() {
      raw.invalidateSize({ pan: false });
      const size = raw.getSize();
      if (!size.x || !size.y) return; // Hidden image/video tabs resize on return.
      const z = coverZoom();
      raw.setMinZoom(z);
      raw.setView([-H / 2, W / 2], z, { animate: false });
    }
    const boundsPixels = bounds => {
      const b = bounds?.getSouthWest ? bounds : L.latLngBounds(bounds);
      const sw = b.getSouthWest(), ne = b.getNorthEast();
      return L.latLngBounds([
        toPixelLatLng(sw), toPixelLatLng(ne),
        toPixelLatLng([sw.lat, ne.lng]), toPixelLatLng([ne.lat, sw.lng])
      ]);
    };
    const map = {
      focusArtwork: home,
      getContainer: () => raw.getContainer(), createPane: n => raw.createPane(n),
      getPane: n => raw.getPane(n), getPanes: () => raw.getPanes(),
      on(events, fn) { raw.on(events, fn); return this; },
      once(events, fn) { raw.once(events, fn); return this; },
      whenReady: fn => raw.whenReady(fn),
      invalidateSize() {
        raw.invalidateSize({ pan: false });
        const size = raw.getSize(), key = `${size.x}:${size.y}`;
        if (!size.x || !size.y) return;
        if (key !== lastSize) { lastSize = key; home(); }
      },
      addLayer(layer) { raw.addLayer(layer); return this; },
      removeLayer(layer) { raw.removeLayer(layer); return this; },
      hasLayer: layer => raw.hasLayer(layer),
      latLngToContainerPoint: p => raw.latLngToContainerPoint(focusedPosition(p)),
      artworkToContainerPoint: p => raw.latLngToContainerPoint([-p.y, p.x]),
      flyToArtwork(p, zoom = 8, options = {}) {
        raw.setView([-p.y, p.x], Math.max(coverZoom(), Math.min(3, zoom - 6)), { ...options, animate: false });
      },
      fitBounds(bounds, options = {}) {
        raw.fitBounds(boundsPixels(bounds), { ...options, maxZoom: Math.min(2, (options.maxZoom || 7) - 6) });
      },
      fitArtworkBounds(points, options = {}) {
        raw.fitBounds(L.latLngBounds(points), { ...options, maxZoom: Math.min(2, (options.maxZoom || 7) - 6) });
      },
      flyTo(p, zoom = 8, options = {}) {
        raw.setView(focusedPosition(p), Math.max(coverZoom(), Math.min(3, zoom - 6)), { ...options, animate: false });
      }
    };
    function wrapLayer(layer) {
      const originalAdd = layer.addTo;
      layer.addTo = function (target) { return originalAdd.call(this, target === map ? raw : target); };
      return layer;
    }
    // Native SVG circles avoid the tiny bounding-box drift of two SVG arc paths
    // at large zoom levels. Keep Leaflet's events, styles and popup handling.
    const exactRenderers = new Map();
    const ExactCircleRenderer = L.SVG.extend({
      _initPath(layer) {
        L.SVG.prototype._initPath.call(this, layer);
        const circle = L.SVG.create("circle");
        for (const attr of layer._path.attributes) circle.setAttribute(attr.name, attr.value);
        layer._path = circle;
      },
      _updateCircle(layer) {
        const point = layer._point;
        layer._path.setAttribute("cx", point.x);
        layer._path.setAttribute("cy", point.y);
        layer._path.setAttribute("r", layer._empty() ? 0 : layer._radius);
      }
    });
    function pointLayer(factory, p, options = {}) {
      const prediction = String(options.className || "").split(/\s+/).includes("prediction-point");
      if (prediction) {
        const pane = options.pane || "overlayPane";
        if (!exactRenderers.has(pane)) exactRenderers.set(pane, new ExactCircleRenderer({ pane }));
        options = { ...options, renderer: exactRenderers.get(pane) };
      }
      let location, locationKey;
      function mappedLocation(value) {
        const g = L.latLng(value);
        const margin = Math.max(24, ((Number(options.radius) || 7.5) + (Number(options.weight) || 0)/2 + 2) / 2**coverZoom());
        const key = `${g.lat}:${g.lng}:${options.artworkCity}:${margin}`;
        if (key !== locationKey) {
          locationKey = key;
          location = predictionPixel(g, options.artworkCity, margin);
        }
        return location;
      }
      const position = value => {
        if (!prediction) return options.artworkPosition
          ? L.latLng(-options.artworkPosition.y, options.artworkPosition.x) : toPixelLatLng(value);
        const v = mappedLocation(value);
        return L.latLng(-v.y, v.x);
      };
      const layer = wrapLayer(factory(position(p), prediction
        ? { ...options, opacity: 1, fillOpacity: 1 } : options));
      if (prediction) {
        const project = layer._project;
        layer._project = function () {
          if (this._map) this._latlng = position(geographic);
          project.call(this);
          if (!this._map) return;
          const target = mappedLocation(geographic);
          const image = artwork.getElement();
          const rect = image.getBoundingClientRect();
          // Convert the image's exact screen center into this SVG's coordinates.
          // Fractional zoom also transforms Leaflet's SVG renderer by a fraction
          // of a pixel, so sharing layer coordinates alone is insufficient.
          const svg = this._renderer._container;
          const matrix = svg.getScreenCTM();
          if (matrix) {
            const screen = new DOMPoint(rect.left + target.x / W * rect.width,
              rect.top + target.y / H * rect.height).matrixTransform(matrix.inverse());
            this._point = L.point(screen.x, screen.y);
          }
          // The PPT circle has radius 10.1 plus a 1px white rim in image pixels.
          // Cover its whole disc at every zoom, including antialiasing edges.
          this._radius = target.kind === "exact"
            ? Math.ceil(Math.max(Number(options.radius) || 7.5,
                11.1 * Math.max(rect.width / W, rect.height / H) + 2))
            : Math.max(0.5, Math.min(Number(options.radius) || 7.5,
                window.HEXI_CORRIDOR_GEOMETRY.clearance(target) * Math.min(rect.width/W, rect.height/H)
                  - (Number(options.weight) || 0)/2 - 1));
          this._path?.setAttribute("data-location-rule", target.kind);
          this._radiusY = this._radius;
          this._updateBounds();
        };
        // Reproject after the image has completed its own zoom/reset handlers.
        // Otherwise a zoom can briefly reuse the image's previous rounded origin.
        const syncToArtwork = () => layer.redraw();
        layer.on("add", () => {
          predictionFocus = { layer, geographic: () => geographic };
          raw.on("zoomend moveend viewreset resize", syncToArtwork);
          artwork.on("load", syncToArtwork);
          syncToArtwork();
        });
        layer.on("remove", () => {
          if (predictionFocus?.layer === layer) predictionFocus = null;
          raw.off("zoomend moveend viewreset resize", syncToArtwork);
          artwork.off("load", syncToArtwork);
        });
      }
      const setter = layer.setLatLng;
      let geographic = L.latLng(p), bindingPopup = false;
      layer.setLatLng = function (v) {
        geographic = L.latLng(v);
        return setter.call(this, position(v));
      };
      // Leaflet popup internals need pixel coordinates; application click callbacks
      // continue receiving the original geographic coordinate used by route cards.
      const on = layer.on, bind = layer.bindPopup;
      layer.on = function (event, handler, context) {
        if (event === "click" && typeof handler === "function" && !bindingPopup) {
          const callback = handler;
          handler = e => callback.call(context || this, { ...e, latlng: geographic });
        }
        return on.call(this, event, handler, context);
      };
      layer.bindPopup = function (...args) {
        bindingPopup = true;
        try { return bind.apply(this, args); } finally { bindingPopup = false; }
      };
      return layer;
    }
    const api = {
      ...L,
      layerGroup: layers => wrapLayer(L.layerGroup(layers)),
      circleMarker: (p, o) => pointLayer(L.circleMarker, p, o),
      marker: (p, o) => pointLayer(L.marker, p, o),
      polyline(points, options) {
        // Route paths already use artwork pixels; never warp their intermediate
        // vertices through geographic snapping intended for city locations.
        if (options?.artworkPixels) return wrapLayer(L.polyline(points, options));
        const layer = wrapLayer(L.polyline(points.map(toPixelLatLng), options));
        const setter = layer.setLatLngs;
        layer.setLatLngs = points => setter.call(layer, points.map(toPixelLatLng));
        return layer;
      },
      geoJSON(data, options = {}) {
        // Keep callback coordinates geographic; point factories do the conversion.
        return wrapLayer(L.geoJSON(data, { ...options, coordsToLatLng: c => L.latLng(c[1], c[0]) }));
      },
      heatLayer(points, options = {}) {
        return wrapLayer(L.layerGroup(points.map(p => api.circleMarker(p, {
          pane: options.pane, radius: (options.radius || 24) * 0.7,
          stroke: false, fillColor: '#e99836', fillOpacity: Math.min(0.24, p[2] * 0.2), interactive: false
        }))));
      }
    };
    home();
    new ResizeObserver(() => map.invalidateSize()).observe(raw.getContainer());
    return { map, L: api };
  };
})();
