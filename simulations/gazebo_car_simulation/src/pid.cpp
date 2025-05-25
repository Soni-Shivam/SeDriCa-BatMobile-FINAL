#include "pid.hpp"

PID::PID(double kp, double ki, double kd)
    : Kp(kp), Ki(ki), Kd(kd), prev_error(0.0), integral(0.0) {}

double PID::compute(double error, double dt) {
    // Update integral term
    integral += error * dt;
    
    // Calculate derivative term
    double derivative = (error - prev_error) / dt;
    
    // Update previous error
    prev_error = error;
    
    // Calculate PID output
    double output = Kp * error + Ki * integral + Kd * derivative;
    
    return output;
} 