#ifndef JPOS_CONTROLLER
#define JPOS_CONTROLLER

#include <RobotController.h>
#include "JPosUserParameters.h"
#include <torch/script.h>
#include <cstring>


class karl_Controller: public RobotController{
  public:
    karl_Controller(): RobotController(), _jpos_ini(cheetah::num_act_joint){
    _jpos_ini.setZero();
    }
    virtual ~karl_Controller(){}

    virtual void initializeController();
    virtual void runController();
    virtual void updateVisualization(){}
    virtual ControlParameters* getUserControlParameters() {
      return &userParameters;
    }
  protected:
    DVec<float> _jpos_ini;

  float _bodyHeight;
  Eigen::Matrix<float, 3, 1> _bodyOri;
  Eigen::Matrix<float, 12, 1> _jointQ;
  Eigen::Matrix<float, 3, 1> _bodyVel;
  Eigen::Matrix<float, 3, 1> _bodyAngularVel;
  Eigen::Matrix<float, 12, 1> _jointQd;
  Eigen::VectorXf _obs;
  int _obsDim;
  Eigen::VectorXf _obsMean, _obsVar;

  std::string _loadPath;
  torch::jit::script::Module _actor;
  std::vector<torch::jit::IValue> _input;
  std::vector<torch::jit::IValue> _action;


  karlUserParameters userParameters;
};

#endif
