from selfdrive.car.interfaces import CarInterfaceBase
from selfdrive.car.ford.interface import CarInterface as LegacyCarInterface
from selfdrive.car.ford_bp.carcontroller import CarController
from selfdrive.car.ford_bp.carstate import CarState


class CarInterface(CarInterfaceBase):
  _get_params = LegacyCarInterface._get_params

  def __init__(self, CP, CarControllerUnused, CarStateUnused):
    super().__init__(CP, CarController, CarState)
