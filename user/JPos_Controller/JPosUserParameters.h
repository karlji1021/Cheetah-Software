#ifndef PROJECT_JPOSUSERPARAMETERS_H
#define PROJECT_JPOSUSERPARAMETERS_H

#include "ControlParameters/ControlParameters.h"

class karlUserParameters : public ControlParameters {
public:
  karlUserParameters()
      : ControlParameters("user-parameters"),
        INIT_PARAMETER(tau_ff),
        INIT_PARAMETER(kp),
        INIT_PARAMETER(kd),
        INIT_PARAMETER(zero),
        INIT_PARAMETER(calibrate),
        INIT_PARAMETER(q_init1),
        INIT_PARAMETER(q_init2),
        INIT_PARAMETER(q_init3),
        INIT_PARAMETER(q_init4)
      {}

  DECLARE_PARAMETER(double, tau_ff);
  DECLARE_PARAMETER(Vec3<double>, kp);
  DECLARE_PARAMETER(Vec3<double>, kd);
  DECLARE_PARAMETER(double, zero);
  DECLARE_PARAMETER(double, calibrate);
  DECLARE_PARAMETER(Vec3<double>, q_init1);
  DECLARE_PARAMETER(Vec3<double>, q_init2);
  DECLARE_PARAMETER(Vec3<double>, q_init3);
  DECLARE_PARAMETER(Vec3<double>, q_init4);
};

#endif //PROJECT_JPOSUSERPARAMETERS_H
