import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.random_dataset_generator import DataType, Distribution, ColumnConfig, DatasetConfig, Dataset

print("模块导入成功")

ds = Dataset(DatasetConfig(
    rows=5,
    columns=[
        ColumnConfig(name='id', dtype=DataType.INTEGER, min_value=1, max_value=100, unique=True),
        ColumnConfig(name='name', dtype=DataType.STRING, min_value=5, max_value=10),
        ColumnConfig(name='email', dtype=DataType.EMAIL),
        ColumnConfig(name='phone', dtype=DataType.PHONE),
        ColumnConfig(name='score', dtype=DataType.FLOAT, min_value=0, max_value=100),
        ColumnConfig(name='birthday', dtype=DataType.DATE, min_value='2000-01-01', max_value='2020-12-31'),
        ColumnConfig(name='active', dtype=DataType.BOOLEAN),
        ColumnConfig(name='city', dtype=DataType.CATEGORICAL, choices=['北京', '上海', '深圳', '杭州', '成都']),
    ]
))

ds.generate()
ds.preview()
