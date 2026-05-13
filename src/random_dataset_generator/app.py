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


ERROR_MSG = {
    "no_columns": "请至少配置一列数据",
    "invalid_rows": "行数必须为正整数",
    "invalid_dtype": "不支持的数据类型: {dtype}",
    "invalid_distribution": "不支持的分布类型: {dist}",
    "categorical_no_choices": "分类类型（categorical）必须提供选项列表",
    "null_ratio_range": "空值率必须在 0~1 之间",
    "column_name_empty": "列名不能为空",
    "invalid_min_max": "最小值不能大于最大值",
    "unsupported_format": "不支持的导出格式: {fmt}",
    "generate_failed": "数据生成失败: {detail}",
}


@app.route("/api/types", methods=["GET"])
def get_types():
    """获取支持的数据类型和分布类型"""
    type_labels = {
        "integer": "整数", "float": "浮点数", "string": "字符串",
        "date": "日期", "datetime": "日期时间", "boolean": "布尔值",
        "categorical": "分类", "email": "邮箱", "phone": "手机号", "uuid": "UUID",
    }
    dist_labels = {
        "uniform": "均匀分布", "normal": "正态分布",
        "exponential": "指数分布", "choice": "选择分布",
    }
    return jsonify({
        "data_types": [{"value": t.value, "label": type_labels.get(t.value, t.name)} for t in DataType],
        "distributions": [{"value": d.value, "label": dist_labels.get(d.value, d.name)} for d in Distribution],
    })


def _parse_columns(columns_cfg):
    """解析前端传来的列配置，返回 ColumnConfig 列表和错误信息"""
    if not columns_cfg:
        return None, ERROR_MSG["no_columns"]

    columns = []
    for i, col in enumerate(columns_cfg, 1):
        name = col.get("name", "").strip()
        if not name:
            return None, f"第 {i} 列: {ERROR_MSG['column_name_empty']}"

        dtype_str = col.get("dtype", "")
        try:
            dtype = DataType(dtype_str)
        except ValueError:
            return None, ERROR_MSG["invalid_dtype"].format(dtype=dtype_str)

        dist_str = col.get("distribution", "uniform")
        try:
            distribution = Distribution(dist_str)
        except ValueError:
            return None, ERROR_MSG["invalid_distribution"].format(dist=dist_str)

        if dtype == DataType.CATEGORICAL and not col.get("choices"):
            return None, f"第 {i} 列（{name}）: {ERROR_MSG['categorical_no_choices']}"

        min_val = col.get("min_value")
        max_val = col.get("max_value")

        # 日期/时间类型保留字符串
        if dtype in (DataType.DATE, DataType.DATETIME):
            min_val = min_val if min_val and str(min_val).strip() else None
            max_val = max_val if max_val and str(max_val).strip() else None
        else:
            # 数值类型转换
            if min_val is not None and min_val != "":
                min_val = float(min_val) if dtype == DataType.FLOAT else int(float(min_val))
            else:
                min_val = None
            if max_val is not None and max_val != "":
                max_val = float(max_val) if dtype == DataType.FLOAT else int(float(max_val))
            else:
                max_val = None

        if min_val is not None and max_val is not None and dtype not in (DataType.DATE, DataType.DATETIME):
            if min_val > max_val:
                return None, f"第 {i} 列（{name}）: {ERROR_MSG['invalid_min_max']}"

        null_ratio = float(col.get("null_ratio", 0))
        if null_ratio < 0 or null_ratio > 1:
            return None, f"第 {i} 列（{name}）: {ERROR_MSG['null_ratio_range']}"

        choices = col.get("choices")
        if choices and isinstance(choices, str):
            choices = [s.strip() for s in choices.split(",") if s.strip()]

        columns.append(ColumnConfig(
            name=name,
            dtype=dtype,
            min_value=min_val,
            max_value=max_val,
            choices=choices,
            pattern=col.get("pattern") or None,
            format=col.get("format") or None,
            null_ratio=null_ratio,
            unique=bool(col.get("unique", False)),
            distribution=distribution,
            mean=float(col["mean"]) if col.get("mean") is not None else None,
            std_dev=float(col["std_dev"]) if col.get("std_dev") is not None else None,
        ))

    return columns, None


@app.route("/api/generate", methods=["POST"])
def generate():
    """根据配置生成数据集"""
    try:
        body = request.json or {}
        rows = body.get("rows", 100)
        if not isinstance(rows, int) or rows <= 0:
            return jsonify({"success": False, "error": ERROR_MSG["invalid_rows"]}), 400

        seed = body.get("seed", None)
        columns, err = _parse_columns(body.get("columns", []))
        if err:
            return jsonify({"success": False, "error": err}), 400

        config = DatasetConfig(rows=rows, columns=columns, seed=seed)
        ds = Dataset(config)
        ds.generate()

        return jsonify({
            "success": True,
            "data": ds.to_dict(),
            "rows": len(ds.data),
            "columns": [col.name for col in columns],
        })
    except Exception as e:
        return jsonify({"success": False, "error": ERROR_MSG["generate_failed"].format(detail=str(e))}), 400


@app.route("/api/export/<fmt>", methods=["POST"])
def export(fmt):
    """导出数据集为 CSV / JSON"""
    try:
        body = request.json or {}
        rows = body.get("rows", 100)
        if not isinstance(rows, int) or rows <= 0:
            return jsonify({"success": False, "error": ERROR_MSG["invalid_rows"]}), 400

        seed = body.get("seed", None)
        columns, err = _parse_columns(body.get("columns", []))
        if err:
            return jsonify({"success": False, "error": err}), 400

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
            return jsonify({"success": False, "error": ERROR_MSG["unsupported_format"].format(fmt=fmt)}), 400

        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name=filename, mimetype=mimetype)

    except Exception as e:
        return jsonify({"success": False, "error": ERROR_MSG["generate_failed"].format(detail=str(e))}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
