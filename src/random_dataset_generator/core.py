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
