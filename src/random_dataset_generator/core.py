"""核心模块 - 数据类型定义和数据集模型"""

from __future__ import annotations

import random
import string
import re
import csv
import json
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, Union, Callable
from pathlib import Path


class DataType(Enum):
    """支持的数据类型枚举"""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    EMAIL = "email"
    PHONE = "phone"
    UUID = "uuid"
    NAME = "name"
    ADDRESS = "address"
    URL = "url"
    IP = "ip"
    COLOR = "color"
    COUNTRY = "country"


class Distribution(Enum):
    """数据分布类型"""
    UNIFORM = "uniform"
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    CHOICE = "choice"


@dataclass
class ColumnConfig:
    """列配置类，定义单列的生成规则"""
    name: str
    dtype: DataType
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[list] = None
    pattern: Optional[str] = None
    format: Optional[str] = None
    null_ratio: float = 0.0
    unique: bool = False
    distribution: Distribution = Distribution.UNIFORM
    mean: Optional[float] = None
    std_dev: Optional[float] = None
    generator: Optional[Callable[[], Any]] = None

    def __post_init__(self):
        if self.null_ratio < 0 or self.null_ratio > 1:
            raise ValueError("null_ratio must be between 0 and 1")
        if self.dtype == DataType.CATEGORICAL and not self.choices:
            raise ValueError("Categorical column requires choices list")


@dataclass
class DatasetConfig:
    """数据集配置类"""
    rows: int = 100
    columns: list[ColumnConfig] = field(default_factory=list)
    seed: Optional[int] = None
    missing_values: Optional[list] = None

    def __post_init__(self):
        if self.rows <= 0:
            raise ValueError("rows must be positive")
        if self.missing_values is None:
            self.missing_values = ["", None, "N/A", "null"]


class DataGenerator:
    """数据生成器基类"""

    @staticmethod
    def generate_integer(config: ColumnConfig) -> int:
        if config.distribution == Distribution.NORMAL:
            mean = config.mean or (config.min_value + config.max_value) / 2 if config.min_value is not None else 0
            std = config.std_dev or 1
            return int(random.gauss(mean, std))
        else:
            min_val = config.min_value if config.min_value is not None else 0
            max_val = config.max_value if config.max_value is not None else 1000
            return random.randint(int(min_val), int(max_val))

    @staticmethod
    def generate_float(config: ColumnConfig) -> float:
        if config.distribution == Distribution.NORMAL:
            mean = config.mean or (config.min_value + config.max_value) / 2 if config.min_value is not None else 0.0
            std = config.std_dev or 1.0
            value = random.gauss(mean, std)
        elif config.distribution == Distribution.EXPONENTIAL:
            scale = config.mean or 1.0
            value = random.expovariate(1 / scale)
        else:
            min_val = float(config.min_value) if config.min_value is not None else 0.0
            max_val = float(config.max_value) if config.max_value is not None else 1000.0
            value = random.uniform(min_val, max_val)

        if config.min_value is not None and value < config.min_value:
            value = config.min_value
        if config.max_value is not None and value > config.max_value:
            value = config.max_value
        return round(value, 2)

    @staticmethod
    def generate_string(config: ColumnConfig) -> str:
        if config.pattern:
            return DataGenerator._generate_from_pattern(config.pattern)
        length = random.randint(config.min_value or 5, config.max_value or 20)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    @staticmethod
    def _generate_from_pattern(pattern: str) -> str:
        result = []
        i = 0
        while i < len(pattern):
            if pattern[i] == '<':
                end = pattern.find('>', i)
                if end != -1:
                    placeholder = pattern[i+1:end]
                    result.append(DataGenerator._expand_placeholder(placeholder))
                    i = end + 1
                else:
                    result.append(pattern[i])
                    i += 1
            else:
                result.append(pattern[i])
                i += 1
        return ''.join(result)

    @staticmethod
    def _expand_placeholder(placeholder: str) -> str:
        if placeholder == 'L':
            return random.choice(string.ascii_uppercase)
        elif placeholder == 'l':
            return random.choice(string.ascii_lowercase)
        elif placeholder == 'd':
            return random.choice(string.digits)
        elif placeholder == 'A':
            return random.choice(string.ascii_letters)
        else:
            return placeholder

    @staticmethod
    def generate_date(config: ColumnConfig) -> str:
        if config.min_value and config.max_value:
            start = datetime.fromisoformat(config.min_value) if isinstance(config.min_value, str) else datetime.combine(config.min_value, datetime.min.time())
            end = datetime.fromisoformat(config.max_value) if isinstance(config.max_value, str) else datetime.combine(config.max_value, datetime.min.time())
        else:
            start = datetime.now() - timedelta(days=365)
            end = datetime.now()
        delta = end - start
        random_days = random.randint(0, delta.days)
        date = start + timedelta(days=random_days)
        return date.strftime(config.format or "%Y-%m-%d")

    @staticmethod
    def generate_datetime(config: ColumnConfig) -> str:
        if config.min_value and config.max_value:
            start = datetime.fromisoformat(config.min_value) if isinstance(config.min_value, str) else datetime.combine(config.min_value, datetime.min.time())
            end = datetime.fromisoformat(config.max_value) if isinstance(config.max_value, str) else datetime.combine(config.max_value, datetime.min.time())
        else:
            start = datetime.now() - timedelta(days=365)
            end = datetime.now()
        delta = end - start
        random_seconds = random.randint(0, int(delta.total_seconds()))
        dt = start + timedelta(seconds=random_seconds)
        return dt.strftime(config.format or "%Y-%m-%d %H:%M:%S")

    @staticmethod
    def generate_boolean(config: ColumnConfig) -> bool:
        return random.choice([True, False])

    @staticmethod
    def generate_categorical(config: ColumnConfig) -> Any:
        return random.choice(config.choices)

    @staticmethod
    def generate_email(config: ColumnConfig) -> str:
        username_length = random.randint(5, 15)
        username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=username_length))
        domains = ["gmail.com", "qq.com", "163.com", "outlook.com", "yahoo.com"]
        return f"{username}@{random.choice(domains)}"

    @staticmethod
    def generate_phone(config: ColumnConfig) -> str:
        if config.pattern:
            return DataGenerator._generate_from_pattern(config.pattern)
        prefixes = ["138", "139", "150", "151", "186", "187", "135", "136"]
        number = ''.join(random.choices(string.digits, k=8))
        return f"{random.choice(prefixes)}{number}"

    @staticmethod
    def generate_uuid(config: ColumnConfig) -> str:
        return f"{random.randint(0, 0xFFFFFFFF):08x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFFFFFFFFFF):012x}"

    # ---- 新增数据类型 ----

    _LAST_NAMES_ZH = ["王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴",
                       "徐", "孙", "胡", "朱", "高", "林", "何", "郭", "马", "罗"]
    _FIRST_NAMES_ZH = ["伟", "芳", "娜", "敏", "静", "强", "磊", "洋", "勇", "军",
                       "杰", "丽", "超", "秀英", "明", "华", "建华", "玉兰", "建国", "英"]
    _LAST_NAMES_EN = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                      "Miller", "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson",
                      "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White"]
    _FIRST_NAMES_EN = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer",
                       "Michael", "Linda", "David", "Elizabeth", "William", "Barbara",
                       "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen"]

    @staticmethod
    def generate_name(config: ColumnConfig) -> str:
        if config.pattern == "en":
            return f"{random.choice(DataGenerator._FIRST_NAMES_EN)} {random.choice(DataGenerator._LAST_NAMES_EN)}"
        last = random.choice(DataGenerator._LAST_NAMES_ZH)
        first_len = random.randint(1, 2)
        first = "".join(random.choice(DataGenerator._FIRST_NAMES_ZH) for _ in range(first_len))
        return last + first

    _STREETS = ["中山路", "解放路", "人民路", "建设路", "和平路", "长安路", "文化路", "学院路",
                "新华路", "朝阳路", "友谊路", "光明路", "幸福路", "民主路", "胜利路"]
    _CITIES_ZH = ["北京市", "上海市", "广州市", "深圳市", "杭州市", "成都市", "武汉市",
                  "南京市", "重庆市", "西安市", "苏州市", "长沙市", "天津市", "郑州市"]

    @staticmethod
    def generate_address(config: ColumnConfig) -> str:
        city = random.choice(DataGenerator._CITIES_ZH)
        street = random.choice(DataGenerator._STREETS)
        number = random.randint(1, 200)
        building = random.randint(1, 30)
        unit = random.randint(1, 6)
        room = random.randint(101, 1205)
        return f"{city}{street}{number}号{building}栋{unit}单元{room}室"

    _DOMAINS = ["example.com", "test.org", "demo.net", "sample.io", "mysite.com",
                "webapp.dev", "platform.co", "service.cn", "cloud.tech", "data.info"]
    _PATHS = ["home", "api", "users", "products", "docs", "search", "dashboard", "settings", "profile", "items"]

    @staticmethod
    def generate_url(config: ColumnConfig) -> str:
        protocol = random.choice(["https", "http"])
        domain = random.choice(DataGenerator._DOMAINS)
        path = random.choice(DataGenerator._PATHS)
        param = random.randint(1, 9999)
        return f"{protocol}://{domain}/{path}/{param}"

    @staticmethod
    def generate_ip(config: ColumnConfig) -> str:
        if config.pattern == "v6":
            groups = [f"{random.randint(0, 0xFFFF):04x}" for _ in range(8)]
            return ":".join(groups)
        return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

    @staticmethod
    def generate_color(config: ColumnConfig) -> str:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    _COUNTRIES_ZH = ["中国", "美国", "日本", "韩国", "英国", "法国", "德国", "意大利", "西班牙",
                     "加拿大", "澳大利亚", "巴西", "印度", "俄罗斯", "墨西哥", "泰国", "越南",
                     "新加坡", "马来西亚", "印度尼西亚"]

    @staticmethod
    def generate_country(config: ColumnConfig) -> str:
        return random.choice(DataGenerator._COUNTRIES_ZH)


class Dataset:
    """数据集类"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data: list[dict] = []
        self._generated_values: dict[str, set] = {}

        if config.seed is not None:
            random.seed(config.seed)

    def generate(self) -> "Dataset":
        """生成完整数据集"""
        self._generated_values = {col.name: set() for col in self.config.columns if col.unique}

        for _ in range(self.config.rows):
            row = self._generate_row()
            self.data.append(row)
        return self

    def _generate_row(self) -> dict[str, Any]:
        row = {}
        for col in self.config.columns:
            value = self._generate_value(col)

            if col.null_ratio > 0 and random.random() < col.null_ratio:
                value = random.choice(self.config.missing_values)

            row[col.name] = value
        return row

    def _generate_value(self, config: ColumnConfig) -> Any:
        if config.generator:
            value = config.generator()
        else:
            generators = {
                DataType.INTEGER: DataGenerator.generate_integer,
                DataType.FLOAT: DataGenerator.generate_float,
                DataType.STRING: DataGenerator.generate_string,
                DataType.DATE: DataGenerator.generate_date,
                DataType.DATETIME: DataGenerator.generate_datetime,
                DataType.BOOLEAN: DataGenerator.generate_boolean,
                DataType.CATEGORICAL: DataGenerator.generate_categorical,
                DataType.EMAIL: DataGenerator.generate_email,
                DataType.PHONE: DataGenerator.generate_phone,
                DataType.UUID: DataGenerator.generate_uuid,
                DataType.NAME: DataGenerator.generate_name,
                DataType.ADDRESS: DataGenerator.generate_address,
                DataType.URL: DataGenerator.generate_url,
                DataType.IP: DataGenerator.generate_ip,
                DataType.COLOR: DataGenerator.generate_color,
                DataType.COUNTRY: DataGenerator.generate_country,
            }
            generator = generators.get(config.dtype)
            if not generator:
                raise ValueError(f"Unsupported data type: {config.dtype}")
            value = generator(config)

        if config.unique:
            max_attempts = 1000
            attempts = 0
            while value in self._generated_values[config.name] and attempts < max_attempts:
                value = generator(config)
                attempts += 1
            self._generated_values[config.name].add(value)

        return value

    def to_dict(self) -> list[dict]:
        """转换为字典列表"""
        return self.data

    def to_dataframe(self):
        """转换为pandas DataFrame（如果可用）"""
        try:
            import pandas as pd
            return pd.DataFrame(self.data)
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")

    def save_csv(self, filepath: Union[str, Path], **kwargs) -> None:
        """保存为CSV文件"""
        with open(filepath, 'w', newline='', encoding=kwargs.get('encoding', 'utf-8')) as f:
            if not self.data:
                return
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)

    def save_json(self, filepath: Union[str, Path], **kwargs) -> None:
        """保存为JSON文件"""
        with open(filepath, 'w', encoding=kwargs.get('encoding', 'utf-8')) as f:
            json.dump(self.data, f, indent=kwargs.get('indent', 2), ensure_ascii=False)

    def save_excel(self, filepath: Union[str, Path], **kwargs) -> None:
        """保存为Excel文件"""
        try:
            import pandas as pd
            df = self.to_dataframe()
            df.to_excel(filepath, index=False, engine=kwargs.get('engine', 'openpyxl'))
        except ImportError:
            raise ImportError("pandas and openpyxl are required for Excel export. Install with: pip install pandas openpyxl")

    def preview(self, n: int = 5) -> "Dataset":
        """预览前n行数据"""
        for i, row in enumerate(self.data[:n]):
            print(f"Row {i + 1}: {row}")
        return self
