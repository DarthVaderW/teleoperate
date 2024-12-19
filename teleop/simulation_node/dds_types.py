from dataclasses import dataclass
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array

@dataclass
class RotationMatrix(IdlStruct, typename="RotationMatrix"):
    data: array[float, 9+16+16+12+12]  # SMPL rotation matrix

@dataclass
class Angle_38(IdlStruct, typename="angle38"):
    data: array[float, 38]     # joint angles

@dataclass
class Angle_14(IdlStruct, typename="angle14"):
    data: array[float, 14]
