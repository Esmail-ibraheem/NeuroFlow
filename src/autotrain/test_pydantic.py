from pydantic import BaseModel, Field
from typing import Optional

class TestParams(BaseModel):
    quantization: Optional[str] = Field("int4", title="Quantization")

params_dict = {"quantization": None}
params = TestParams(**params_dict)
print(f"Quantization (None passed): {params.quantization}")

params_dict_missing = {}
params_missing = TestParams(**params_dict_missing)
print(f"Quantization (Missing): {params_missing.quantization}")

params_dict_none_str = {"quantization": "none"}
params_none_str = TestParams(**params_dict_none_str)
print(f"Quantization ('none' passed): {params_none_str.quantization}")
