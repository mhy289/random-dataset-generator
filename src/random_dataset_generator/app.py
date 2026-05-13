"""Web 应用 - Flask 后端 API"""

import io
import json
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory

# 支持直接运行 (python app.py) 和模块运行 (python -m src.random_dataset_generator.app)
try:
    from .core import DataType, Distribution, ColumnConfig, DatasetConfig, Dataset
except ImportError:
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from src.random_dataset_generator import DataType, Distribution, ColumnConfig, DatasetConfig, Dataset

app = Flask(__name__, static_folder=None)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


@app.route("/api/types", methods=["GET"])
def get_types():
    """获取支持的数据类型和分布类型"""
    return jsonify({
        "data_types": [{"value": t.value, "label": t.name} for t in DataType],
        "distributions": [{"value": d.value, "label": d.name} for d in Distribution],
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    """根据配置生成数据集"""
    try:
        body = request.json
        rows = body.get("rows", 100)
        seed = body.get("seed", None)
        columns_cfg = body.get("columns", [])

        columns = []
        for col in columns_cfg:
            dtype = DataType(col["dtype"])
            distribution = Distribution(col.get("distribution", "uniform"))

            min_val = col.get("min_value")
            max_val = col.get("max_value")
            if min_val is not None:
                min_val = float(min_val) if dtype in (DataType.FLOAT, DataType.EXPONENTIAL) else int(float(min_val))
            if max_val is not None:
                max_val = float(max_val) if dtype in (DataType.FLOAT, DataType.EXPONENTIAL) else int(float(max_val))

            columns.append(ColumnConfig(
                name=col["name"],
                dtype=dtype,
                min_value=min_val,
                max_value=max_val,
                choices=col.get("choices"),
                pattern=col.get("pattern"),
                format=col.get("format"),
                null_ratio=float(col.get("null_ratio", 0)),
                unique=bool(col.get("unique", False)),
                distribution=distribution,
                mean=float(col["mean"]) if col.get("mean") is not None else None,
                std_dev=float(col["std_dev"]) if col.get("std_dev") is not None else None,
            ))

        config = DatasetConfig(rows=rows, columns=columns, seed=seed)
        ds = Dataset(config)
        ds.generate()

        return jsonify({
            "success": True,
            "data": ds.to_dict(),
            "rows": len(ds.data),
            "columns": [col["name"] for col in columns_cfg],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/export/<fmt>", methods=["POST"])
def export(fmt):
    """导出数据集为 CSV / JSON / Excel"""
    try:
        body = request.json
        rows = body.get("rows", 100)
        seed = body.get("seed", None)
        columns_cfg = body.get("columns", [])

        columns = []
        for col in columns_cfg:
            dtype = DataType(col["dtype"])
            distribution = Distribution(col.get("distribution", "uniform"))

            min_val = col.get("min_value")
            max_val = col.get("max_value")
            if min_val is not None:
                min_val = float(min_val) if dtype in (DataType.FLOAT,) else int(float(min_val))
            if max_val is not None:
                max_val = float(max_val) if dtype in (DataType.FLOAT,) else int(float(max_val))

            columns.append(ColumnConfig(
                name=col["name"],
                dtype=dtype,
                min_value=min_val,
                max_value=max_val,
                choices=col.get("choices"),
                pattern=col.get("pattern"),
                format=col.get("format"),
                null_ratio=float(col.get("null_ratio", 0)),
                unique=bool(col.get("unique", False)),
                distribution=distribution,
                mean=float(col["mean"]) if col.get("mean") is not None else None,
                std_dev=float(col["std_dev"]) if col.get("std_dev") is not None else None,
            ))

        config = DatasetConfig(rows=rows, columns=columns, seed=seed)
        ds = Dataset(config)
        ds.generate()

        buf = io.BytesIO()

        if fmt == "csv":
            text_buf = io.StringIO()
            if ds.data:
                import csv
                writer = csv.DictWriter(text_buf, fieldnames=ds.data[0].keys())
                writer.writeheader()
                writer.writerows(ds.data)
            buf.write(text_buf.getvalue().encode("utf-8-sig"))
            mimetype = "text/csv"
            filename = "dataset.csv"
        elif fmt == "json":
            buf.write(json.dumps(ds.to_dict(), indent=2, ensure_ascii=False).encode("utf-8"))
            mimetype = "application/json"
            filename = "dataset.json"
        else:
            return jsonify({"success": False, "error": f"Unsupported format: {fmt}"}), 400

        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name=filename, mimetype=mimetype)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
