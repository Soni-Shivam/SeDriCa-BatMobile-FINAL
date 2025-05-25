#ifndef PID_HPP
#define PID_HPP

class PID {
public:
    PID(double kp, double ki, double kd);
    double compute(double error, double dt);

private:
    double Kp, Ki, Kd;
    double prev_error, integral;
};

#endif // PID_HPP
