from cereal import car
from common.logger import sLogger
from common.numpy_fast import clip, interp
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_std_steer_angle_limits
from selfdrive.car.ford.fordcan import create_acc_msg, create_acc_ui_msg, create_button_msg, create_lat_ctl_msg, \
  create_lat_ctl2_msg, create_lka_msg, create_lkas_ui_msg
from selfdrive.car.ford.values import CANBUS, CANFD_CARS, CarControllerParams
from selfdrive.modeld.constants import T_IDXS

# Limit lateral acceleration for CAN-FD platforms to avoid aggressive curvature on banked roads.
EARTH_G = 9.81
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees
MAX_LATERAL_ACCEL = 3.0 - (EARTH_G * AVERAGE_ROAD_ROLL)
# Small right-bias when lane lines are not trusted to avoid centering on unlined roads.
RIGHT_EDGE_BIAS_CURVATURE = 0.0003
PATH_OFFSET_LOOKAHEAD = 0.2
PATH_OFFSET_MAX = 2.0
PATH_ANGLE_MAX = 0.5
LANE_CONFIDENCE_BP = [0.6, 0.8]

LongCtrlState = car.CarControl.Actuators.LongControlState
VisualAlert = car.CarControl.HUDControl.VisualAlert


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, is_canfd, bias=0.0):
  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                           current_curvature + CarControllerParams.CURVATURE_ERROR)

  # Curvature rate limit after driver torque limit
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, CarControllerParams)

  apply_curvature += bias
  apply_curvature = clip(apply_curvature, -CarControllerParams.CURVATURE_MAX, CarControllerParams.CURVATURE_MAX)

  if is_canfd:
    # Conservative max lateral accel limit for CAN-FD platforms.
    curvature_accel_limit = MAX_LATERAL_ACCEL / (max(v_ego_raw, 1.0) ** 2)
    apply_curvature = clip(apply_curvature, -curvature_accel_limit, curvature_accel_limit)

  return apply_curvature


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.VM = VM
    self.packer = CANPacker(dbc_name)
    self.frame = 0

    self.apply_curvature_last = 0
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False

  def update(self, CC, sm, CS, now_nanos):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(create_button_msg(self.packer, CS.buttons_stock_values, cancel=True))
      can_sends.append(create_button_msg(self.packer, CS.buttons_stock_values, cancel=True, bus=CANBUS.main))
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(create_button_msg(self.packer, CS.buttons_stock_values, resume=True))
      can_sends.append(create_button_msg(self.packer, CS.buttons_stock_values, resume=True, bus=CANBUS.main))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(create_button_msg(self.packer, CS.buttons_stock_values, tja_toggle=True))

    ### lateral control ###
    # send steer msg at 20Hz
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      if CC.latActive:
        # apply rate limits, curvature error limit, and clip to signal range
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
        lane_line_bias = 0.0
        if CC.latActive and not sm['lateralPlan'].useLaneLines:
          lane_line_bias = -RIGHT_EDGE_BIAS_CURVATURE
        desired_curvature_rate = 0.0
        path_offset = 0.0
        path_angle = 0.0
        try:
          desired_curvature_rate = float(sm['controlsState'].desiredCurvatureRate)
        except Exception:
          desired_curvature_rate = 0.0
        desired_curvature_rate = clip(desired_curvature_rate, -0.001023, 0.001023)

        try:
          model = sm['modelV2']
          path_offset_position = interp(PATH_OFFSET_LOOKAHEAD, T_IDXS, model.position.y)
          path_offset_lanelines = (model.laneLines[1].y[0] + model.laneLines[2].y[0]) / 2
          laneline_confidence = min(model.laneLineProbs[1], model.laneLineProbs[2])
          laneline_scale = interp(laneline_confidence, LANE_CONFIDENCE_BP, [0.0, 1.0])
          if not sm['lateralPlan'].useLaneLines:
            laneline_scale = 0.0
          path_offset = (path_offset_position * (1.0 - laneline_scale)) + (path_offset_lanelines * laneline_scale)
          if sm['lateralPlan'].laneChangeState != 0:
            path_offset = 0.0
        except Exception:
          path_offset = 0.0
        path_offset = clip(path_offset, -PATH_OFFSET_MAX, PATH_OFFSET_MAX)

        try:
          path_angle = float(sm['lateralPlan'].psis[0])
        except Exception:
          path_angle = 0.0
        path_angle = clip(path_angle, -PATH_ANGLE_MAX, PATH_ANGLE_MAX)
        apply_curvature = apply_ford_curvature_limits(actuators.curvature, self.apply_curvature_last, current_curvature,
                                                      CS.out.vEgoRaw, self.CP.carFingerprint in CANFD_CARS,
                                                      bias=lane_line_bias)
      else:
        apply_curvature = 0.
        desired_curvature_rate = 0.0
        path_offset = 0.0
        path_angle = 0.0

      self.apply_curvature_last = apply_curvature

      if self.CP.carFingerprint in CANFD_CARS:
        # TODO: extended mode
        mode = 1 if CC.latActive else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(create_lat_ctl2_msg(self.packer, mode, -path_offset, -path_angle,
                                             -apply_curvature, -desired_curvature_rate, counter))
      else:
        can_sends.append(create_lat_ctl_msg(self.packer, CC.latActive, -path_offset, -path_angle,
                                            -apply_curvature, -desired_curvature_rate))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      can_sends.append(create_lka_msg(self.packer))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      # Both gas and accel are in m/s^2, accel is used solely for braking
      accel = clip(actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX)
      gas = accel
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS

      stopping = CC.actuators.longControlState == LongCtrlState.stopping
      can_sends.append(create_acc_msg(self.packer, CC.longActive, gas, accel, stopping))

    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # send lkas ui msg at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(create_lkas_ui_msg(self.packer, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))
    # send acc ui msg at 5Hz or if ui state changes
    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      can_sends.append(create_acc_ui_msg(self.packer, self.CP, main_on, CC.latActive,
                                         CS.out.cruiseState.standstill, hud_control,
                                         CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert

    new_actuators = actuators.copy()
    new_actuators.curvature = self.apply_curvature_last

    self.frame += 1
    if (self.frame % 10) == 0:
      # Send a debug heartbeat so the Android UI doesn't show "System Unresponsive".
      sLogger.Send("0ford cc ok")
    return new_actuators, can_sends
