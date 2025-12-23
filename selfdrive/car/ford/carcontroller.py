import math
from cereal import car
from common.logger import sLogger
from common.numpy_fast import clip, interp
from common.realtime import DT_CTRL
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_hysteresis, apply_std_steer_angle_limits
from selfdrive.car.ford.fordcan import create_acc_msg, create_acc_ui_msg, create_button_msg, create_lat_ctl_msg, \
  create_lat_ctl2_msg, create_lka_msg, create_lkas_ui_msg
from selfdrive.car.ford.values import CANBUS, CANFD_CARS, CarControllerParams

# Limit lateral acceleration for CAN-FD platforms to avoid aggressive curvature on banked roads.
EARTH_G = 9.81
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees
MAX_LATERAL_ACCEL = 3.0 - (EARTH_G * AVERAGE_ROAD_ROLL)
# Small right-bias when lane lines are not trusted to avoid centering on unlined roads.
RIGHT_EDGE_BIAS_CURVATURE = 0.0003

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


def anti_overshoot(apply_curvature, apply_curvature_last, v_ego):
  diff = 0.1
  tau = 5.0
  dt = DT_CTRL * CarControllerParams.STEER_STEP
  alpha = 1.0 - math.exp(-dt / tau)

  lataccel = apply_curvature * (v_ego ** 2)
  last_lataccel = apply_curvature_last * (v_ego ** 2)
  last_lataccel = apply_hysteresis(lataccel, last_lataccel, diff)
  last_lataccel = alpha * lataccel + (1.0 - alpha) * last_lataccel

  output_curvature = last_lataccel / (max(v_ego, 1.0) ** 2)
  return float(interp(v_ego, [5.0, 10.0], [apply_curvature, output_curvature]))


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.VM = VM
    self.packer = CANPacker(dbc_name)
    self.frame = 0

    self.apply_curvature_last = 0
    self.anti_overshoot_curvature_last = 0.0
    self.post_reset_ramp_active = False
    self.reset_steering_last = False
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
        requested_curvature = apply_ford_curvature_limits(actuators.curvature, self.apply_curvature_last, current_curvature,
                                                          CS.out.vEgoRaw, self.CP.carFingerprint in CANFD_CARS,
                                                          bias=lane_line_bias)
        reset_steering = CS.steeringPressed
        if reset_steering:
          self.post_reset_ramp_active = False
          self.anti_overshoot_curvature_last = 0.0
          apply_curvature = 0.0
        else:
          if self.reset_steering_last and not reset_steering:
            self.post_reset_ramp_active = True
            self.apply_curvature_last = 0.0

          if self.post_reset_ramp_active:
            # Ramp back in after driver steering to avoid snap back.
            apply_curvature = apply_std_steer_angle_limits(requested_curvature, self.apply_curvature_last,
                                                           CS.out.vEgoRaw, CarControllerParams)
            if abs(apply_curvature - requested_curvature) < max(0.001, 0.1 * abs(requested_curvature)):
              self.post_reset_ramp_active = False
          else:
            self.anti_overshoot_curvature_last = anti_overshoot(requested_curvature,
                                                                self.anti_overshoot_curvature_last,
                                                                CS.out.vEgoRaw)
            apply_curvature = self.anti_overshoot_curvature_last
        self.reset_steering_last = reset_steering
      else:
        apply_curvature = 0.
        self.post_reset_ramp_active = False

      self.apply_curvature_last = apply_curvature

      if self.CP.carFingerprint in CANFD_CARS:
        # TODO: extended mode
        mode = 1 if CC.latActive else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(create_lat_ctl2_msg(self.packer, mode, 0., 0., -apply_curvature, 0., counter))
      else:
        can_sends.append(create_lat_ctl_msg(self.packer, CC.latActive, 0., 0., -apply_curvature, 0.))

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
