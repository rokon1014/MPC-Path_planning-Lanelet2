import casadi


def kinematic_model(states, control, vehicle, TIME_STEP):
    """vehicle kinematic model"""
    _x = states[0]
    _y = states[1]
    _yaw = states[2]

    _delta = control[0]
    _vx = control[1]

    _beta = casadi.atan2(vehicle.lr * casadi.tan(_delta), (vehicle.lf + vehicle.lr))

    _dx = _vx * casadi.cos(_yaw + _beta)
    _dy = _vx * casadi.sin(_yaw + _beta)
    _dyaw = (_vx * casadi.cos(_beta) * casadi.tan(_delta)) / (vehicle.lf + vehicle.lr)

    _x = _x + _dx * TIME_STEP
    _y = _y + _dy * TIME_STEP
    _yaw = _yaw + _dyaw * TIME_STEP
    _st = [_x, _y, _yaw]

    return _st
