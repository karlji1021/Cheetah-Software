#include "JPos_Controller.hpp"
#include <random>


void JPos_Controller::initializeController() {
  _loadPath = "/home/user/raisim_workspace/Cheetah-Software/actor_model/actor_1000.pt";
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    _actor = torch::jit::load(_loadPath);
  }
  catch (const c10::Error& e) {
    std::cerr << "!!!!Error loading the model!!!!\n";
  }
  _obsDim = 34;

  _obsMean.setZero(_obsDim);
  _obsVar.setZero(_obsDim);
  std::string in_line;
  std::ifstream obsMean_file("/home/user/raisim_workspace/Cheetah-Software/actor_model/mean1000.csv");
  std::ifstream obsVariance_file("/home/user/raisim_workspace/Cheetah-Software/actor_model/var1000.csv");
  if(obsMean_file.is_open()) {
    for(int i = 0; i < _obsMean.size(); i++){
      std::getline(obsMean_file, in_line);
      _obsMean(i) = std::stod(in_line);
    }
  }
  if(obsVariance_file.is_open()) {
    for(int i = 0; i < _obsVar.size(); i++){
      std::getline(obsVariance_file, in_line);
      _obsVar(i) = std::stod(in_line);
    }
  }
  obsMean_file.close();
  obsVariance_file.close();
}

void JPos_Controller::runController(){
  Eigen::Matrix<float, 12, 1> qInitVec;
  Vec3<float> qInitVec1, qInitVec2, qInitVec3, qInitVec4;
  Mat3<float> kpMat;
  Mat3<float> kdMat;
  //kpMat << 20, 0, 0, 0, 20, 0, 0, 0, 20;
  //kdMat << 2.1, 0, 0, 0, 2.1, 0, 0, 0, 2.1;
  qInitVec1 << userParameters.q_init1[0], userParameters.q_init1[1], userParameters.q_init1[2];
  qInitVec2 << userParameters.q_init2[0], userParameters.q_init2[1], userParameters.q_init2[2];
  qInitVec3 << userParameters.q_init3[0], userParameters.q_init3[1], userParameters.q_init3[2];
  qInitVec4 << userParameters.q_init4[0], userParameters.q_init4[1], userParameters.q_init4[2];
  qInitVec << qInitVec1, qInitVec2, qInitVec3, qInitVec4;
  kpMat << userParameters.kp[0], 0, 0, 0,  userParameters.kp[1], 0, 0, 0,  userParameters.kp[2];
  kdMat <<  userParameters.kd[0], 0, 0, 0, userParameters.kd[1], 0, 0, 0, userParameters.kd[2];



  /// TODO: check each variable is appropriate.
  _bodyHeight = _stateEstimate->position(2);  // body height 1
  _bodyOri << _stateEstimate->rpy;  _bodyOri(2) += 1; // roll pitch yaw 3, raisim has default yaw value of 1.
  _jointQ << _legController->datas[0].q, _legController->datas[1].q, _legController->datas[2].q, _legController->datas[3].q;  // joint angles 3 3 3 3 = 12
  _bodyVel << _stateEstimate->vWorld;  // velocity 3
  _bodyAngularVel << _stateEstimate->omegaWorld;  // angular velocity 3
  _jointQd << _legController->datas[0].qd, _legController->datas[0].qd, _legController->datas[0].qd, _legController->datas[0].qd;  // joint velocity 3 3 3 3 = 12

  _obs.setZero(_obsDim);
  _obs << _bodyHeight,
    _bodyOri,
    _jointQ,
    _bodyVel,
    _bodyAngularVel,
    _jointQd;
  for(int i = 0; i < _obs.size(); i++) {
    _obs(i) = (_obs(i) - _obsMean(i)) / std::sqrt(_obsVar(i) + 1e-8);
    if(_obs(i) > 10) _obs(i) = 10.0;
    if(_obs(i) < -10) _obs(i) = -10.0;
  }


  _input.push_back(torch::jit::IValue(torch::from_blob(_obs.data(), {_obs.cols(),_obs.rows()}).clone()));
  std::cout << torch::from_blob(_obs.data(), {_obs.cols(),_obs.rows()}) << std::endl;
  torch::Tensor action_tensor = _actor.forward(_input).toTensor();  // action of Tensor type
  float* action_f = action_tensor.data_ptr<float>();  // action of float pointer type
  Eigen::Map<Eigen::MatrixXf> action_eigen(action_f, action_tensor.size(0), action_tensor.size(1));  // action of MatrixXf type
  _input.pop_back();

//  std::cout << "action_eigen size: " << action_eigen.size() << std::endl;


  static int iter(0);
  ++iter;

  if(iter < 10){
    for(int leg(0); leg<4; ++leg){
      for(int jidx(0); jidx<3; ++jidx){
        _jpos_ini[3*leg+jidx] = _legController->datas[leg].q[jidx];
      }
    }
  }

  _legController->_maxTorque = 18;
  _legController->_legsEnabled = true;

  if(userParameters.calibrate > 0.4) {
    _legController->_calibrateEncoders = userParameters.calibrate;
  } else {
    if(userParameters.zero > 0.5) {
      _legController->_zeroEncoders = true;
    } else {
      _legController->_zeroEncoders = false;

      for(int leg(0); leg<4; ++leg){
        for(int jidx(0); jidx<3; ++jidx){
          std::cout << _legController->datas[leg].q[jidx];
//          _legController->commands[leg].qDes[jidx] = action_eigen(leg*3 + jidx);
          _legController->commands[leg].qDes[jidx] = qInitVec(leg*3 + jidx);
          _legController->commands[leg].qdDes[jidx] = 0.;
          _legController->commands[leg].tauFeedForward[jidx] = userParameters.tau_ff;
        }
        _legController->commands[leg].kpJoint = kpMat;
        _legController->commands[leg].kdJoint = kdMat;
      }
      std::cout << std::endl;
    }
  }



  //if(iter%200 ==0){
    //printf("value 1, 2: %f, %f\n", userParameters.testValue, userParameters.testValue2);
  //}


}
