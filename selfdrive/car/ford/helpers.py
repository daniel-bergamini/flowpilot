from cereal import car
from common.realtime import DT_CTRL
from system.swaglog import cloudlog


def hysteresis(current_value, old_value, target: float, delta: float):
  if target < current_value < min(target + delta, 0):
    result = old_value
  elif current_value <= target:
    result = 1
  elif current_value >= min(target + delta, 0):
    result = 0

  return result


def actuators_calc(cc_self, brake):  # cc_self is the CarController object (self)
  ts = cc_self.frame * DT_CTRL

  brake_actuate = hysteresis(brake, cc_self.brake_actuate_last, cc_self.brake_actuator_activate, cc_self.brake_actuator_release_delta)
  cc_self.brake_actuate_last = brake_actuate

  precharge_actuate = hysteresis(
    brake, cc_self.precharge_actuate_last, (cc_self.brake_actuator_activate + cc_self.precharge_actuator_target_delta),
    (cc_self.brake_actuator_release_delta - cc_self.precharge_actuator_target_delta)
  )

  if precharge_actuate and not cc_self.precharge_actuate_last:
    cc_self.precharge_actuate_ts = ts
  elif not precharge_actuate:
    cc_self.precharge_actuate_ts = 0

  if (
    precharge_actuate
    and not brake_actuate
    and cc_self.precharge_actuate_ts > 0
    and brake > cc_self.brake_actuator_activate
    and (ts - cc_self.precharge_actuate_ts) > (200 * DT_CTRL)
  ):
    precharge_actuate = False

  cc_self.precharge_actuate_last = precharge_actuate
  cloudlog.debug("actuators_calc: %s %s %s", brake, precharge_actuate, brake_actuate)

  return precharge_actuate, brake_actuate


def _event_names_from_state(dm_state):
  names = set()
  if dm_state is None:
    return names
  events = getattr(dm_state, "events", None)
  if events is None:
    return names

  for e in events:
    name_val = getattr(e, "name", None)
    if name_val is None:
      continue
    if hasattr(name_val, "name"):
      names.add(name_val.name)
    elif isinstance(name_val, int):
      try:
        names.add(car.CarEvent.EventName(name_val).name)
      except Exception:
        continue
    else:
      names.add(str(name_val))
  return names


def get_dm_state(d_state, main_on):
  e = d_state.split("/")
  if main_on:
    en = e[0]
    et = e[-1]
  else:
    en = "none"
    et = "none"
  return en, et


def _driver_state_from_events(dm_state):
  names = _event_names_from_state(dm_state)
  # prioritize prompt > active > pre
  if "promptDriverUnresponsive" in names:
    return "promptDriverUnresponsive"
  if "promptDriverDistracted" in names:
    return "promptDriverDistracted"
  if "driverUnresponsive" in names:
    return "driverUnresponsive"
  if "driverDistracted" in names:
    return "driverDistracted"
  if "preDriverUnresponsive" in names:
    return "preDriverUnresponsive"
  if "preDriverDistracted" in names:
    return "preDriverDistracted"
  return "none"


def compute_dm_msg_values(dm_state, hud_control, send_hands_free_cluster_msg, main, standstill=False, disable_state="none"):
  tja_msg = 0
  tja_warn = 0
  hands = 0

  if dm_state is None:
    driverState, disableState = "none", "none"
  elif hasattr(dm_state, "alertType"):
    driverState, disableState = get_dm_state(dm_state.alertType, main)
  else:
    driverState = _driver_state_from_events(dm_state)
    disableState = disable_state

  if send_hands_free_cluster_msg:
    if disableState == "noEntry":
      tja_msg = 1  # Lane Centering Assist not available
    elif (driverState in ("driverDistracted", "driverUnresponsive") or disableState in ("softDisable", "immediateDisable")):
      tja_warn = 3  # Resume Control
    elif disableState == "userDisable":
      tja_warn = 1  # Cancelled
    elif driverState == "preDriverDistracted":
      hands = 1  # Keep Hands on Steering Wheel (no chime)
    elif driverState == "promptDriverDistracted":
      if not standstill:
        hands = 2  # Keep Hands on Steering Wheel (chime)
      else:
        hands = 1
    elif driverState == "preDriverUnresponsive":
      hands = 1
    elif driverState == "promptDriverUnresponsive":
      if not standstill:
        hands = 2
      else:
        hands = 1
    elif hud_control.leftLaneDepart:
      tja_warn = 5  # Left Lane Departure (chime)
    elif hud_control.rightLaneDepart:
      tja_warn = 4  # Right Lane Departure (chime)
    else:
      tja_warn = 0
  else:
    if disableState == "noEntry":
      tja_msg = 1  # Lane Centering Assist not available
    elif (driverState in ("driverDistracted", "driverUnresponsive") or disableState in ("softDisable", "immediateDisable")):
      tja_warn = 3  # Resume Control
    elif disableState == "userDisable":
      tja_warn = 1  # Cancelled
    elif driverState in ("preDriverDistracted", "preDriverUnresponsive"):
      hands = 1  # Keep Hands on Steering Wheel (no chime)
    elif driverState in ("promptDriverDistracted", "promptDriverUnresponsive"):
      if not standstill:
        hands = 2  # Keep Hands on Steering Wheel (chime)
      else:
        hands = 1
    else:
      tja_warn = 0
  return tja_msg, tja_warn, hands
