import math
import time

from cereal import car
from common.params import Params
from common.pid import PIDController
from common.logger import sLogger
from common.numpy_fast import clip, interp
from common.realtime import DT_CTRL
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_hysteresis, apply_std_steer_angle_limits
from selfdrive.car.ford.fordcan import create_acc_msg, create_acc_ui_msg, create_button_msg, create_lat_ctl_msg, \
  create_lat_ctl2_msg, create_lka_msg, create_lkas_ui_msg
from selfdrive.car.ford.helpers import compute_dm_msg_values
from selfdrive.car.ford.values import CANBUS, CANFD_CARS, CarControllerParams, FordConfig
from selfdrive.modeld.constants import T_IDXS

# Limit lateral acceleration for CAN-FD platforms to avoid aggressive curvature on banked roads.
EARTH_G = 9.81
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees
MAX_LATERAL_ACCEL = 3.0 - (EARTH_G * AVERAGE_ROAD_ROLL)
# Small right-bias when lane lines are not trusted to avoid centering on unlined roads.
RIGHT_EDGE_BIAS_CURVATURE = 0.0003
LANE_LINE_BIAS_SCALE = 0.0001
PATH_OFFSET_LOOKAHEAD = 0.2
PATH_OFFSET_MAX = 2.0
PATH_ANGLE_MAX = 0.5
LANE_CONFIDENCE_BP = [0.6, 0.8]
LANE_CHANGE_FACTOR_BP = [4.4, 40.23]
LANE_CHANGE_FACTOR_V = [0.95, 0.85]
LANE_WIDTH_TOLERANCE_BP = [3.75, 4.25]
LANE_WIDTH_TOLERANCE_V = [0.81, 0.59]
LC_PID_SPEED_BP = [0.0, 9.0, 15.0]
LC_PID_SPEED_V = [0.0, 0.0, 1.0]
LC_PATH_ANGLE_ROC_BP = [5.0, 15.0, 25.0]
LC_PATH_ANGLE_ROC_V = [0.003, 0.0015, 0.002]
POST_LANE_CHANGE_FRAMES = 160
POST_LANE_CHANGE_MAX_PATH_ANGLE_CHANGE = 0.00125
POST_LANE_CHANGE_MAX_PATH_OFFSET_CHANGE = 0.00125
POST_LANE_CHANGE_MAX_CURVATURE_RATE_CHANGE = 0.0001

LongCtrlState = car.CarControl.Actuators.LongControlState
VisualAlert = car.CarControl.HUDControl.VisualAlert


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, is_canfd,
                                max_lateral_accel, bias=0.0):
  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                           current_curvature + CarControllerParams.CURVATURE_ERROR)

  # Curvature rate limit after driver torque limit
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, CarControllerParams)

  apply_curvature += bias
  apply_curvature = clip(apply_curvature, -CarControllerParams.CURVATURE_MAX, CarControllerParams.CURVATURE_MAX)

  if is_canfd and max_lateral_accel is not None:
    # Conservative max lateral accel limit for CAN-FD platforms.
    curvature_accel_limit = max_lateral_accel / (max(v_ego_raw, 1.0) ** 2)
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
    self.send_hands_free_cluster_msg = FordConfig.BLUECRUISE_CLUSTER_PRESENT
    self.tja_msg = 0
    self.tja_warn = 0
    self.hands = 0
    self.params = Params()
    self.precision_type = 0
    self._last_precision_update = 0.0
    self.max_lateral_accel = MAX_LATERAL_ACCEL
    self.lane_line_bias = RIGHT_EDGE_BIAS_CURVATURE
    self.hud_enhancements = True
    self.steer_rate_profile = 1
    self.use_bp_lane_positioning = False
    self.enable_lane_positioning = False
    self.enable_lanefull_mode = False
    self.custom_path_offset = 0.0
    self.path_offset_lookup_time = PATH_OFFSET_LOOKAHEAD
    self.lc_pid_gain_ui = 0.0
    self.lc_pid_controller = PIDController(k_p=0.25, k_i=0.05, rate=20)
    self.lc_path_angle_reset_counter = 0
    self.lc_path_angle_reset_duration = 1.5
    self.path_angle_last = 0.0
    self.lane_change = False
    self.lane_change_last = False
    self.post_lane_change_active = False
    self.post_lane_change_timer = 0
    self.pre_lane_change_values = {
      'path_angle': 0.0,
      'path_offset': 0.0,
      'desired_curvature_rate': 0.0,
    }

  def _update_precision_type(self):
    now = time.monotonic()
    if now - self._last_precision_update < 1.0:
      return
    self._last_precision_update = now
    try:
      raw = self.params.get("FordLatCtlPrecisionMode")
      if raw is None:
        raw = ""
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace").strip()
      value = int(raw)
      if value in (0, 1):
        self.precision_type = value
    except (ValueError, TypeError):
      pass
    try:
      raw = self.params.get("FordCanfdMaxLateralAccel")
      if raw is None:
        raw = ""
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace").strip()
      value = float(raw)
      if value > 0.0:
        self.max_lateral_accel = value
      else:
        self.max_lateral_accel = None
    except (ValueError, TypeError):
      pass
    try:
      raw = self.params.get("FordLaneLineBias")
      if raw is None:
        raw = ""
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace").strip()
      value = int(raw)
      self.lane_line_bias = value * LANE_LINE_BIAS_SCALE
    except (ValueError, TypeError):
      pass
    try:
      raw = self.params.get("FordSteerRateProfile")
      if raw is None:
        raw = ""
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace").strip()
      value = int(raw)
      if value in (0, 1) and value != self.steer_rate_profile:
        self.steer_rate_profile = value
        if value == 0:
          CarControllerParams.ANGLE_RATE_LIMIT_UP = CarControllerParams.LEGACY_ANGLE_RATE_LIMIT_UP
          CarControllerParams.ANGLE_RATE_LIMIT_DOWN = CarControllerParams.LEGACY_ANGLE_RATE_LIMIT_DOWN
        else:
          CarControllerParams.ANGLE_RATE_LIMIT_UP = CarControllerParams.BLUEPILOT_ANGLE_RATE_LIMIT_UP
          CarControllerParams.ANGLE_RATE_LIMIT_DOWN = CarControllerParams.BLUEPILOT_ANGLE_RATE_LIMIT_DOWN
    except (ValueError, TypeError):
      pass
    try:
      raw = self.params.get("FordBpHudEnhancements")
      if raw is None:
        raw = ""
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace").strip()
      value = int(raw)
      if value in (0, 1):
        self.hud_enhancements = bool(value)
    except (ValueError, TypeError):
      pass
    try:
      raw = self.params.get("FordEnableBpLanePositioning")
      if raw is None:
        raw = ""
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace").strip()
      value = int(raw)
      if value in (0, 1):
        self.use_bp_lane_positioning = bool(value)
    except (ValueError, TypeError):
      pass
    try:
      raw = self.params.get("FordLatTuningEnableLanePositioning")
      if raw is None:
        raw = ""
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace").strip()
      value = int(raw)
      if value in (0, 1):
        self.enable_lane_positioning = bool(value)
    except (ValueError, TypeError):
      pass
    try:
      raw = self.params.get("FordLatTuningEnableLanefullMode")
      if raw is None:
        raw = ""
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace").strip()
      value = int(raw)
      if value in (0, 1):
        self.enable_lanefull_mode = bool(value)
    except (ValueError, TypeError):
      pass
    try:
      raw = self.params.get("FordLatTuningCustomPathOffset")
      if raw is None:
        raw = ""
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace").strip()
      value = float(raw)
      if -0.5 <= value <= 0.5:
        self.custom_path_offset = value
    except (ValueError, TypeError):
      pass
    try:
      raw = self.params.get("FordLatTuningLCPIDGainUI")
      if raw is None:
        raw = ""
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace").strip()
      value = float(raw)
      if value >= 0.0:
        self.lc_pid_gain_ui = value
    except (ValueError, TypeError):
      pass
    try:
      raw = self.params.get("FordLatTuningPathOffsetLookupTime")
      if raw is None:
        raw = ""
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace").strip()
      value = float(raw)
      if 0.0 <= value <= 0.8:
        self.path_offset_lookup_time = value
    except (ValueError, TypeError):
      pass

  def _handle_post_lane_change_transition(self, path_angle, path_offset, desired_curvature_rate):
    if self.lane_change_last and not self.lane_change:
      self.post_lane_change_active = True
      self.post_lane_change_timer = 0
      self.pre_lane_change_values = {
        'path_angle': 0.0,
        'path_offset': 0.0,
        'desired_curvature_rate': 0.0,
      }

    self.lane_change_last = self.lane_change

    if self.post_lane_change_active:
      self.post_lane_change_timer += 1
      new_path_angle = clip(
        path_angle,
        self.pre_lane_change_values['path_angle'] - POST_LANE_CHANGE_MAX_PATH_ANGLE_CHANGE,
        self.pre_lane_change_values['path_angle'] + POST_LANE_CHANGE_MAX_PATH_ANGLE_CHANGE,
      )
      new_path_offset = clip(
        path_offset,
        self.pre_lane_change_values['path_offset'] - POST_LANE_CHANGE_MAX_PATH_OFFSET_CHANGE,
        self.pre_lane_change_values['path_offset'] + POST_LANE_CHANGE_MAX_PATH_OFFSET_CHANGE,
      )
      new_curvature_rate = clip(
        desired_curvature_rate,
        self.pre_lane_change_values['desired_curvature_rate'] - POST_LANE_CHANGE_MAX_CURVATURE_RATE_CHANGE,
        self.pre_lane_change_values['desired_curvature_rate'] + POST_LANE_CHANGE_MAX_CURVATURE_RATE_CHANGE,
      )
      self.pre_lane_change_values = {
        'path_angle': new_path_angle,
        'path_offset': new_path_offset,
        'desired_curvature_rate': new_curvature_rate,
      }
      if self.post_lane_change_timer >= POST_LANE_CHANGE_FRAMES:
        self.post_lane_change_active = False
      return new_path_angle, new_path_offset, new_curvature_rate

    return path_angle, path_offset, desired_curvature_rate

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
      self._update_precision_type()
      if CC.latActive:
        # apply rate limits, curvature error limit, and clip to signal range
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
        lane_line_bias = 0.0
        if CC.latActive and not sm['lateralPlan'].useLaneLines:
          lane_line_bias = -self.lane_line_bias
        desired_curvature_rate = 0.0
        path_offset = 0.0
        path_angle = 0.0
        try:
          desired_curvature_rate = float(sm['controlsState'].desiredCurvatureRate)
        except Exception:
          desired_curvature_rate = 0.0
        desired_curvature_rate = clip(desired_curvature_rate, -0.001023, 0.001023)

        reset_steering = CS.out.steeringPressed
        lane_change_active = sm['lateralPlan'].laneChangeState != 0
        self.lane_change = lane_change_active
        if lane_change_active:
          desired_curvature_rate = 0.0

        if self.use_bp_lane_positioning:
          try:
            model = sm['modelV2']
            path_offset_position = interp(self.path_offset_lookup_time, T_IDXS, model.position.y)
            path_offset_lanelines = (model.laneLines[1].y[0] + model.laneLines[2].y[0]) / 2
            laneline_width = model.laneLines[2].y[0] + (-model.laneLines[1].y[0])
            laneline_width_tolerance = interp(laneline_width, LANE_WIDTH_TOLERANCE_BP, LANE_WIDTH_TOLERANCE_V)
            laneline_confidence = min(model.laneLineProbs[1], model.laneLineProbs[2], laneline_width_tolerance)
            if not self.enable_lanefull_mode:
              laneline_confidence = 0.0
            laneline_scale = interp(laneline_confidence, LANE_CONFIDENCE_BP, [0.0, 1.0])
            path_offset = (path_offset_position * (1.0 - laneline_scale)) + (path_offset_lanelines * laneline_scale)
            path_offset += self.custom_path_offset
            if lane_change_active:
              path_offset = 0.0
          except Exception:
            path_offset = 0.0
          path_offset = clip(path_offset, -PATH_OFFSET_MAX, PATH_OFFSET_MAX)

          path_offset_error = path_offset * (self.lc_pid_gain_ui / 100.0)
          lc_pid_speed_factor = interp(CS.out.vEgoRaw, LC_PID_SPEED_BP, LC_PID_SPEED_V)
          path_offset_error_adj = path_offset_error * lc_pid_speed_factor
          if not self.enable_lane_positioning:
            path_offset_error_adj = 0.0
            self.lc_pid_controller.reset()

          path_angle_low_c = self.lc_pid_controller.update(path_offset_error_adj)
          if not self.enable_lane_positioning:
            path_angle_low_c = 0.0
          if reset_steering:
            path_angle_low_c = 0.0

          path_angle_roc = interp(abs(CS.out.vEgoRaw), LC_PATH_ANGLE_ROC_BP, LC_PATH_ANGLE_ROC_V)
          path_angle_low_c = clip(path_angle_low_c, self.path_angle_last - path_angle_roc, self.path_angle_last + path_angle_roc)

          if reset_steering:
            self.lc_path_angle_reset_counter += 1
          else:
            self.lc_path_angle_reset_counter = 0
          if self.lc_path_angle_reset_counter > self.lc_path_angle_reset_duration * 20:
            self.lc_pid_controller.reset()

          path_angle = path_angle_low_c

          path_angle, path_offset, desired_curvature_rate = self._handle_post_lane_change_transition(
            path_angle, path_offset, desired_curvature_rate
          )
          if reset_steering:
            path_angle = 0.0

          desired_curvature_rate = clip(desired_curvature_rate, -0.001023, 0.001023)
          path_offset = clip(path_offset, -PATH_OFFSET_MAX, PATH_OFFSET_MAX)
          path_angle = clip(path_angle, -PATH_ANGLE_MAX, PATH_ANGLE_MAX)

          # avoid sending path_offset when path_angle is used for centering
          path_offset = 0.0
        else:
          try:
            model = sm['modelV2']
            path_offset_position = interp(PATH_OFFSET_LOOKAHEAD, T_IDXS, model.position.y)
            path_offset_lanelines = (model.laneLines[1].y[0] + model.laneLines[2].y[0]) / 2
            laneline_confidence = min(model.laneLineProbs[1], model.laneLineProbs[2])
            laneline_scale = interp(laneline_confidence, LANE_CONFIDENCE_BP, [0.0, 1.0])
            if not sm['lateralPlan'].useLaneLines:
              laneline_scale = 0.0
            path_offset = (path_offset_position * (1.0 - laneline_scale)) + (path_offset_lanelines * laneline_scale)
            if lane_change_active:
              path_offset = 0.0
          except Exception:
            path_offset = 0.0
          path_offset = clip(path_offset, -PATH_OFFSET_MAX, PATH_OFFSET_MAX)

          try:
            path_angle = float(sm['lateralPlan'].psis[0])
          except Exception:
            path_angle = 0.0
          path_angle = clip(path_angle, -PATH_ANGLE_MAX, PATH_ANGLE_MAX)

        self.path_angle_last = path_angle

        requested_curvature = actuators.curvature
        if lane_change_active:
          lane_change_factor = interp(CS.out.vEgoRaw, LANE_CHANGE_FACTOR_BP, LANE_CHANGE_FACTOR_V)
          requested_curvature *= lane_change_factor

        requested_curvature = apply_ford_curvature_limits(requested_curvature, self.apply_curvature_last, current_curvature,
                                                          CS.out.vEgoRaw, self.CP.carFingerprint in CANFD_CARS,
                                                          self.max_lateral_accel, bias=lane_line_bias)
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
        desired_curvature_rate = 0.0
        path_offset = 0.0
        path_angle = 0.0
        self.post_reset_ramp_active = False

      self.apply_curvature_last = apply_curvature

      if self.CP.carFingerprint in CANFD_CARS:
        # TODO: extended mode
        mode = 1 if CC.latActive else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(create_lat_ctl2_msg(self.packer, mode, self.precision_type, -path_offset, -path_angle,
                                             -apply_curvature, -desired_curvature_rate, counter))
      else:
        can_sends.append(create_lat_ctl_msg(self.packer, CC.latActive, self.precision_type, -path_offset, -path_angle,
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
    if self.hud_enhancements:
      dm_state = None
      try:
        dm_state = sm['driverMonitoringState']
      except Exception:
        dm_state = None
      self.tja_msg, self.tja_warn, self.hands = compute_dm_msg_values(
        dm_state, hud_control, self.send_hands_free_cluster_msg, main_on, CS.out.cruiseState.standstill
      )
      if steer_alert:
        self.hands = 1
      elif not self.send_hands_free_cluster_msg:
        self.hands = 0
    else:
      self.hands = 1 if steer_alert else 0
      self.tja_warn = CS.acc_tja_status_stock_values.get("TjaWarn_D_Rq", 0)
      self.tja_msg = CS.acc_tja_status_stock_values.get("TjaMsgTxt_D_Dsply", 0)
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # send lkas ui msg at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(create_lkas_ui_msg(self.packer, main_on, CC.latActive, self.hands, hud_control, CS.lkas_status_stock_values))
    # send acc ui msg at 5Hz or if ui state changes
    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      send_hands_free = self.send_hands_free_cluster_msg if self.hud_enhancements else False
      can_sends.append(create_acc_ui_msg(self.packer, self.CP, main_on, CC.latActive,
                                         CS.out.cruiseState.standstill, hud_control,
                                         CS.acc_tja_status_stock_values, send_hands_free,
                                         self.tja_warn, self.tja_msg,
                                         use_legacy_status=not self.hud_enhancements))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert

    new_actuators = actuators.copy()
    new_actuators.curvature = self.apply_curvature_last

    self.frame += 1
    return new_actuators, can_sends
