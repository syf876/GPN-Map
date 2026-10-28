import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any



ROOT = Path(__file__).parent
POEMS_MASTER = ROOT / "poems_master.csv"

EMOTION_COLORS = {
    "EMO_SORROWFUL": "#7f1d1d",
    "EMO_BOLD": "#dc2626",
    "EMO_HOMESICK": "#2563eb",
    "EMO_MELANCHOLY": "#64748b",
    "EMO_LEISURELY": "#16a34a",
    "EMO_INDIGNANT": "#9333ea",
}

IMAGERY_ICONS = {
    "IMG_FRONTIER": "shield",
    "IMG_DESERT": "dunes",
    "IMG_PASS": "gate",
    "IMG_HOMESICK": "home",
    "IMG_NATURE": "leaf",
    "IMG_FEAST": "cup",
}

DEFAULT_EMOTION_COLOR = "#475569"
DEFAULT_IMAGERY_ICON = "poem"


def load_place_coordinates(master_csv: Path = POEMS_MASTER) -> dict[str, dict[str, Any]]:
    buckets: dict[str, Counter] = defaultdict(Counter)
    with master_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            place = row.get("location", "").strip()
            lat = row.get("lat", "").strip()
            lon = row.get("lon", "").strip()
            if not place or not lat or not lon:
                continue
            try:
                key = (float(lon), float(lat))
            except ValueError:
                continue
            buckets[place][key] += 1

    coords = {}
    for place, counter in buckets.items():
        (lon, lat), count = counter.most_common(1)[0]
        coords[place] = {
            "place": place,
            "lon": lon,
            "lat": lat,
            "source": "poems_master.csv",
            "support": count,
        }
    return coords


def prediction_to_geojson(
    prediction: dict[str, Any],
    coords_by_place: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    coords_by_place = coords_by_place or load_place_coordinates()
    place = prediction["place"]
    place_label = place.get("label") or place.get("id")
    coords = coords_by_place.get(place_label)

    emotion = prediction["emotion"]
    emotion_color = EMOTION_COLORS.get(emotion["id"], DEFAULT_EMOTION_COLOR)
    imagery_labels = prediction["imagery"].get("labels", [])
    imagery_icons = [
        {
            "id": item["id"],
            "label": item["label"],
            "icon": IMAGERY_ICONS.get(item["id"], DEFAULT_IMAGERY_ICON),
            "confidence": item["confidence"],
        }
        for item in imagery_labels
    ]
    primary_icon = imagery_icons[0]["icon"] if imagery_icons else DEFAULT_IMAGERY_ICON

    popup = (
        f"<strong>{prediction['text']}</strong><br/>"
        f"地点：{place_label} ({place['confidence']:.4f})<br/>"
        f"情绪：{emotion['label']} ({emotion['confidence']:.4f})<br/>"
        f"意象：{', '.join(item['label'] for item in imagery_icons) or '无'}"
    )

    feature = {
        "type": "Feature",
        "geometry": None,
        "properties": {
            "text": prediction["text"],
            "dynasty": prediction["dynasty"],
            "place": {
                **place,
                "coordinates": coords,
            },
            "imagery": {
                **prediction["imagery"],
                "icons": imagery_icons,
                "primary_icon": primary_icon,
            },
            "emotion": {
                **emotion,
                "color": emotion_color,
            },
            "confidence": prediction["confidence"],
            "popup": popup,
        },
    }
    if coords:
        feature["geometry"] = {
            "type": "Point",
            "coordinates": [coords["lon"], coords["lat"]],
        }

    return {
        "type": "FeatureCollection",
        "features": [feature],
        "metadata": {
            "format": "geojson",
            "coordinate_source": "poems_master.csv",
            "model": prediction.get("model", {}),
        },
    }


def main():
    from inference_round2 import Round2PoetryPredictor
    parser = argparse.ArgumentParser(description="Round2 推理结果转 GeoJSON")
    parser.add_argument("text", help="输入一句诗")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imagery-threshold", type=float, default=0.5)
    args = parser.parse_args()

    predictor = Round2PoetryPredictor(device=args.device)
    prediction = predictor.predict(args.text, imagery_threshold=args.imagery_threshold)
    geojson = prediction_to_geojson(prediction)
    print(json.dumps(geojson, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
